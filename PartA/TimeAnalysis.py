from datetime import datetime
import matplotlib.pyplot as plt; plt.rcdefaults()
import pyspark
import numpy as np

sc = pyspark.SparkContext()

# load the transactions
lines = sc.textFile("/data/bitcoin/transactions.csv")

# extract header
header = lines.first()
# remove header
lines = lines.filter(lambda line: line != header)

# Transactions Line:
#    0         1        2        3            4
# tx_hash, blockhash, time, tx_in_count, tx_out_count


# function to filter out bad lines
def is_good_line(line):
    try:
        fields = line.split(',')
        if len(fields) != 5:
            return False
        else:
            return True

    except:
        return False

# function to convert timestamp to format "YYYY-MM"
def unix_to_date(timestamp):
    ts = int(timestamp)
    return datetime.fromtimestamp(ts).strftime('%Y-%m')

# function to filter out the transactions outside our timeframe
def in_time(line):
    date = line.split(',')[2]
    startTime = "2009-01"
    myDate = unix_to_date(date)
    if (myDate >= startTime):
        return True
    else:
        return False


# filter out bad lines
# filter out transactions outside our timeframe
# map through the transactions and yield ("YYYY-MM", 1)
# get the sum of the values for each key in the reducer
# sort by earliest date
filtered_transactions = lines.filter(is_good_line) \
    .filter(in_time) \
    .map(lambda line: (unix_to_date(line.split(',')[2]), 1)) \
    .reduceByKey(lambda x, y: (x+y)) \
    .sortBy(lambda x: x[0])


months = []
num_of_transactions = []

# save the results in a text file
filtered_transactions.saveAsTextFile('outBitcoin/TimeAnalysis')

# collect the results to create a plot
for pair in filtered_transactions.collect():
    print(pair)
    myString = "{} ".format(pair[0])
    months.append(myString)
    num_of_transactions.append(pair[1])

y_pos = np.arange(len(months))

# create a plot with the dates on the x axis and number of transactions on the y axis
plt.bar(y_pos, num_of_transactions, align='center', alpha=0.5)
plt.xticks(y_pos, months)
plt.xticks(rotation=75,size=7)
plt.ylabel('Number of transactions')
plt.title('Number of Transactions per Month and Year')
plt.tight_layout()
plt.show()
