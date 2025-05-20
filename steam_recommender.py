from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_replace, split, when, col, array, concat, concat_ws
from pyspark.ml.feature import BucketedRandomProjectionLSH, VectorAssembler, MinMaxScaler, Tokenizer, StopWordsRemover, CountVectorizer

spark = SparkSession.builder.appName("SteamRecommender").config('spark.driver.memory','8g').config('spark.executor.memory', '8g').getOrCreate()
df = spark.read.csv("steam.csv", header=True, inferSchema=True)

df1 = df
df1 = df1.filter(df1['english'] == 1)
df1 = df1.drop(*['release_date', 'english','platforms','required_age', 'average_playtime', 'median_playtime', 'genres', 'achievements'])

columns_dict={'appid':'int', 'name':'string', 'developer':'string','publisher':'string','categories':'string',
              'steamspy_tags':'string','positive_ratings':'int','negative_ratings':'int','owners':'string', 'price':'double'
              }

for column, type in columns_dict.items():
    df1 = df1.withColumn(column, col(column).cast(type))
df1 = df1.withColumn('developer', split(regexp_replace('developer', ' ',''), ';'))
df1 = df1.withColumn('publisher', split(regexp_replace('publisher', ' ',''), ';'))
df1 = df1.withColumn(
    "owners",
    when(col("owners").contains("-"),
         ((split(col("owners"), "-").getItem(0).cast("int") +
           split(col("owners"), "-").getItem(1).cast("int")) / 2).cast("int"))
    .otherwise(col("owners").cast("int"))
)
df1 = df1.withColumn('categories', split(regexp_replace(regexp_replace('categories', '-', ''), ' ', ''), ';'))
df1 = df1.withColumn('steamspy_tags', split(regexp_replace(regexp_replace('steamspy_tags', '-', ''), ' ', ''), ';'))
df1 = df1.withColumn('name_list', array(col('name')))

# tokenizing and assembling columns
numerical_columns = ['owners', 'price']
ratings_df = df1.select('positive_ratings','negative_ratings')

assembler = VectorAssembler(inputCols=numerical_columns, outputCol="num_features", handleInvalid='skip')
assembled = assembler.transform(df1)

scaler = MinMaxScaler(inputCol="num_features", outputCol="scaled_features")
scaler_model = scaler.fit(assembled)
scaled_df = scaler_model.transform(assembled)

# prepare the tags
scaled_df = scaled_df.withColumn('tags', concat(col('name_list'),col('developer'), col('publisher'), col('categories'), col('steamspy_tags'))).withColumn('tags', concat_ws(' ', col('tags')))