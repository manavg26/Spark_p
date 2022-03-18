from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from pyspark.sql.types import StructType,StructField,IntegerType,StringType,LongType
import sys
import codecs

spark=SparkSession.builder.appName("recommend").master("local[*]").getOrCreate()

def cos_compute(data):
    formula=data.withColumn("xx",func.col("rating1")*func.col("rating1"))\
                .withColumn("yy",func.col("rating2")*func.col("rating2"))\
                .withColumn("xy",func.col("rating1")*func.col("rating2"))
    score=formula.groupBy("movie1","movie2").agg(
        func.sum(func.col("xy")).alias("numerator"),\
        (func.sqrt(func.sum(func.col("xx")))*func.sqrt(func.sum(func.col("yy")))).alias("denominator"),\
        func.count(func.col("xy")).alias("numPairs"))

    result=score.withColumn("score",func.when(func.col("denominator")!=0, func.col("numerator")/func.col("denominator"))\
            .otherwise(0)).select("movie1", "movie2", "score", "numPairs")
    return result
def getName(id):
    name=movie_name.filter(func.col("movieId")==id).select("name").collect()[0]
    return name.name
def get_genre(id):
    gen=movie_name.filter(func.col("movieId")==id)\
        .select("unknown","Action","Adventure","Animation","Children's","Comedy","Crime",\
                "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery",\
                "Romance","Sci-Fi","Thriller","War","Western").collect()[0]
    genre_num=[]
    for i in range(len(gen)):
        if(gen[i]==1):
            genre_num.append(i)
    return genre_num
def genre_name():
    dict={}
    with codecs.open("C:/SparkCourse/ml-100k/u.GENRE", "r", errors='ignore') as f:
        for line in f:
            fields=line.split("|")
            dict[int(fields[1])]=str(fields[0])
    return dict
def print_genre(i):
    return namedict.value[i]

namedict=spark.sparkContext.broadcast(genre_name())
schema=StructType([
    StructField("userId",IntegerType(),True),
    StructField("movieId",IntegerType(),True),
    StructField("rating",IntegerType(),True),
    StructField("time",LongType(),True)
    ])
nameSchema=StructType([
    StructField("movieId",IntegerType(),True),
    StructField("name",StringType(),True),
    StructField("date",StringType(),True),
    StructField("link",StringType(),True),
    StructField("empty",StringType(),True),
    StructField("unknown",IntegerType(),True),
    StructField("Action",IntegerType(),True),
    StructField("Adventure",IntegerType(),True),
    StructField("Animation",IntegerType(),True),
    StructField("Children's",IntegerType(),True),
    StructField("Comedy",IntegerType(),True),
    StructField("Crime",IntegerType(),True),
    StructField("Documentary",IntegerType(),True),
    StructField("Drama",IntegerType(),True),
    StructField("Fantasy",IntegerType(),True),
    StructField("Film-Noir",IntegerType(),True),
    StructField("Horror",IntegerType(),True),
    StructField("Musical",IntegerType(),True),
    StructField("Mystery",IntegerType(),True),
    StructField("Romance",IntegerType(),True),
    StructField("Sci-Fi",IntegerType(),True),
    StructField("Thriller",IntegerType(),True),
    StructField("War",IntegerType(),True),
    StructField("Western",IntegerType(),True)
    
    ])
movie=spark.read.schema(schema).option("sep","\t").csv("file:///SparkCourse/ml-100k/u.data")
movie_name=spark.read.schema(nameSchema).option("charset", "ISO-8859-1").option("sep","|").csv("file:///SparkCourse/ml-100k/u.item")
# movie_genre=spark.read.schema(genreSchema).option("charset", "ISO-8859-1").option("sep","|").csv("file:///SparkCourse/ml-100k/u.item")
ratings=movie.filter(func.col("rating")>2).select("userId","movieId","rating")
movie_pair=ratings.alias("rating1").join(ratings.alias("rating2"),\
    (func.col("rating1.userId")==func.col("rating2.userId")) \
    & (func.col("rating1.movieId")<func.col("rating2.movieId"))).select(func.col("rating1.movieId").alias("movie1"),\
    func.col("rating2.movieId").alias("movie2"),func.col("rating1.rating").alias("rating1"),\
    func.col("rating2.rating").alias("rating2"))

similar=cos_compute(movie_pair).cache()
# similar.show()
if (len(sys.argv)>1):
    Movieid=int(sys.argv[1])
    num=get_genre(Movieid)
    scoreThreshold = 0.97
    coOccurrenceThreshold = 70.0
    res=similar.filter(((func.col("movie1")==Movieid)|(func.col("movie2")==Movieid))&\
                        (func.col("score")>scoreThreshold)&(func.col("numPairs")>coOccurrenceThreshold)).sort(func.col("score").desc())
    results=res.take(50)
    print(str(getName(Movieid))+" is of genre:",sep=",",end=" ")
    for i in num:
        print(print_genre(i),end=" ")
    print("\n If you loved "+str(getName(Movieid))+", you would like: ")
    i=0
    for r in results:
        storeId=r.movie1
        if(storeId==Movieid):
            storeId=r.movie2
        gen_num=get_genre(storeId)
        intersection_len=len(list(set(num)&set(gen_num)))
        if (intersection_len>0):
            i+=1
            print(str(getName(storeId))+" score: "+str(r.score)+" Strength: "+str(r.numPairs))
        if(i>=10):
            break

spark.stop()