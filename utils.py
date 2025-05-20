from pyspark.sql import SparkSession
from pyspark.ml.feature import BucketedRandomProjectionLSH, VectorAssembler, MinMaxScaler, Tokenizer, StopWordsRemover, CountVectorizer
from pyspark.sql.functions import regexp_replace, split, when, col, array, concat, concat_ws, lower


def load_model():
    spark = SparkSession.builder.appName("SteamRecommender").config('spark.driver.memory','8g').config('spark.executor.memory', '8g').getOrCreate()

    final_df, lsh_model = build_model(spark)
    return spark, final_df, lsh_model


def run_cli():
    spark, final_df, lsh_model = load_model()

    game = input('Enter the name of a game: ')
    query_df = get_game(final_df, game)
    
    if query_df is None:
        print(f'Game "{game}" not found.')
    else:
        recommendations = recommend(query_df, final_df, lsh_model, game)
        if not recommendations:
            print("No similar games found.")
        else:
            print("Recommended games:")
            for name in recommendations:
                print(name)
    
    spark.stop()


def build_model(spark):
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

    # preparing the tags
    scaled_df = scaled_df.withColumn('tags', concat(col('name_list'),col('developer'), col('publisher'), col('categories'), col('steamspy_tags'))).withColumn('tags', concat_ws(' ', col('tags')))

    tokenizer = Tokenizer(inputCol="tags", outputCol="words")
    words_df = tokenizer.transform(scaled_df)

    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    filtered_df = remover.transform(words_df)

    cv = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=3000, minDF=5)
    model = cv.fit(filtered_df)
    vectorized_df = model.transform(filtered_df)

    final_assembler = VectorAssembler(
        inputCols=['scaled_features', 'features'],
        outputCol='final_features'
    )
    final_df = final_assembler.transform(vectorized_df)

    lsh = BucketedRandomProjectionLSH(inputCol="final_features", outputCol="hashes", bucketLength=2.0, numHashTables=5)
    lsh_model = lsh.fit(final_df)

    return final_df, lsh_model


def get_game(df, game_name):
      """
      Searches for the game name to see if it exists in the database and returns the features of said game.
      """
      query_df = df.filter(lower(df['name']) == game_name.lower()).select('final_features')
      return query_df if not query_df.rdd.isEmpty() else None


def recommend(query_df, final_df, lsh_model, game_name, top_n=5):
        
        if query_df is None or query_df.rdd.isEmpty():
              return 'Game is not recognized (try again!)'

        similar = lsh_model.approxSimilarityJoin(
                datasetA=query_df,
                datasetB=final_df,
                threshold=float('inf'),
                distCol='distance'
        )

        results = (
        similar
        .filter(similar.datasetB["name"] != game_name)
        .orderBy("distance")
        .select(similar.datasetB["name"], "distance")
        .limit(top_n)
        .collect()
    )

        return [row["datasetB.name"] for row in results]