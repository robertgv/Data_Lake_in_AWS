import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, monotonically_increasing_id
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format, dayofweek
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, TimestampType, IntegerType

# Read the configuration file
config = configparser.ConfigParser()
config.read('dl.cfg')

# Read the AWS credentials
os.environ["AWS_ACCESS_KEY_ID"] = config['AWS']['AWS_ACCESS_KEY_ID']
os.environ["AWS_SECRET_ACCESS_KEY"] = config['AWS']['AWS_SECRET_ACCESS_KEY']

# Read the input and output of the data
INPUT_DATA = config['MODE']['INPUT_DATA']
OUTPUT_DATA = config['MODE']['OUTPUT_DATA']

# Define the schema for the JSON song data files
song_data_schema = StructType([
    StructField('num_songs', IntegerType(), True),
    StructField('artist_id', StringType(), False),
    StructField('artist_latitude', DoubleType(), True),
    StructField('artist_longitude', DoubleType(), True),
    StructField('artist_location', StringType(), True),
    StructField('artist_name', StringType(), True),
    StructField('song_id', StringType(), False),
    StructField('title', StringType(), True),
    StructField('duration', DoubleType(), True),
    StructField('year', IntegerType(), True)
])

# Define the schema for the JSON log data files
log_data_schema = StructType([
    StructField('artist', StringType(), True),
    StructField('auth', StringType(), True),
    StructField('firstName', StringType(), True),
    StructField('gender', StringType(), True),
    StructField('itemInSession', LongType(), True),
    StructField('lastName', StringType(), True),
    StructField('length', DoubleType(), True),
    StructField('level', StringType(), True),
    StructField('location', StringType(), True),
    StructField('method', StringType(), True),
    StructField('page', StringType(), True),
    StructField('registration', DoubleType(), True),
    StructField('sessionId', LongType(), False),
    StructField('song', StringType(), True),
    StructField('status', LongType(), True),
    StructField('ts', LongType(), True),
    StructField('userAgent', StringType(), True),
    StructField('userId', StringType(), False),
])


def create_spark_session():
    """
    Get or create a new Spark session to process data
    
    Keywords arguments:
    * NA
    
    Output:
    * spark -- A Spark session
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark):
    """
    Process the song files stored in 'INPUT_DATA' and generates the songs and 
    artists tables in 'OUTPUT_DATA'
    
    Key arguments:
    * spark -- A Spark session
    
    Output:
    * NA
    """
    
    # get filepath to song data file
    song_data = INPUT_DATA + 'song_data/*/*/*/*.json'
    
    # read song data file
    df = spark.read.json(song_data, schema=song_data_schema)

    # extract columns to create songs table
    songs_table = df.select('song_id', 'title', 'artist_id', 'year', 'duration')
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy('year', 'artist_id').parquet(OUTPUT_DATA + 'songs')

    # extract columns to create artists table
    artists_table = df.select('artist_id', 
                              col('artist_name').alias('name'), 
                              col('artist_location').alias('location'),
                              col('artist_latitude').alias('latitude'), 
                              col('artist_longitude').alias('longitude'))
    
    # write artists table to parquet files
    artists_table.write.parquet(OUTPUT_DATA + 'artists')


def process_log_data(spark):
    """
    Process the log files stored in 'INPUT_DATA' and generates the users, time and
    songplays tables in 'OUTPUT_DATA'
    
    Key arguments:
    * spark -- A Spark session
    
    Output:
    * NA
    """
    
    # get filepath to log data file
    #log_data = INPUT_DATA + 'log_data/*/*/*.json' # <-- UNCOMMENT if you are executing in AWS
    log_data = INPUT_DATA + 'log_data/*.json'      # <-- UNCOMMENT if you are executing locally

    # read log data file
    df = spark.read.json(log_data, schema=log_data_schema)
    
    # filter by actions for song plays
    df = df.filter(col('page') == 'NextSong')

    # extract columns for users table    
    users_table = df.select(col('userId').alias('user_id'),
                            col('firstName').alias('first_name'),
                            col('lastName').alias('last_name'),
                            'gender',
                            'level')
    
    # write users table to parquet files
    users_table.write.parquet(OUTPUT_DATA + 'users')

    @udf(TimestampType())
    def get_timestamp(ts):
        return datetime.fromtimestamp(ts / 1000.0)
    
    # create timestamp column from original timestamp column
    df = df.withColumn('start_time', get_timestamp('ts'))
    
    @udf(StringType())
    def get_datetime(ts):
        return datetime.fromtimestamp(ts / 1000.0)\
                       .strftime('%Y-%m-%d %H:%M:%S')
    
    # create datetime column from original timestamp column
    df = df.withColumn('datetime', get_datetime('ts'))
    
    # extract columns to create time table
    time_table = df.select(col('start_time').alias('start_time'),
                           hour(col('start_time')).alias('hour'),
                           dayofmonth(col('start_time')).alias('day'),
                           weekofyear(col('start_time')).alias('week'), 
                           month(col('start_time')).alias('month'),
                           year(col('start_time')).alias('year'),
                           dayofweek(col('start_time')).alias('weekday'))
    
    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy('year','month').parquet(OUTPUT_DATA + 'time')

    # read in song data to use for songplays table
    song_data = INPUT_DATA + 'song_data/*/*/*/*.json'
    
    # read song data file
    song_df = spark.read.json(song_data, schema=song_data_schema)

    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = song_df.join(df, (song_df.artist_name==df.artist) & \
                                       (song_df.title==df.song))
    
    songplays_table = songplays_table.withColumn('songplay_id', monotonically_increasing_id())
    
    songplays_table = songplays_table.select('songplay_id',
                                             year(col('start_time')).alias('year'),
                                             month(col('start_time')).alias('month'),
                                             'start_time',
                                             col('userId').alias('user_id'),
                                             'level',
                                             'song_id',
                                             'artist_id',
                                             col('sessionId').alias('session_id'),
                                             'location',
                                             col('userAgent').alias('user_agent'))
    
    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.partitionBy('year','month').parquet(OUTPUT_DATA + 'songplays')


def main():
    """
    Process the data songs and logs files stored in 'INPUT_DATA' and 
    generates the songplays, users, songs, artists and time tables 
    in 'OUTPUT_DATA'
    """
    
    # Create Spark session
    spark = create_spark_session()
    
    # Process the song data files
    process_song_data(spark)
    
    # Process the log data files
    process_log_data(spark)


if __name__ == "__main__":
    main()
