from pyspark import SparkContext
import matplotlib.pyplot as plt; plt.rcdefaults()

sc = SparkContext()


wikileaksKey = "{1HB5XMLmzFVj8ALj6mfBsbifRoD4miY36v}"

# function to filter out bad lines
def is_vin_good_line(line):
    try:
        fields = line.split(',')
        if len(fields) != 3:
            return False
        else:
            return True

    except:
        return False
# function to filter out bad lines
def is_vout_good_line(line):
    try:
        fields = line.split(',')
        if len(fields) != 4:
            return False
        else:
            return True

    except:
        return False

# function to filter out transactions not related to wikileaks and bad lines
def is_to_wikileaks(line):
    try:
        fields = line.split(',')
        if fields[3] == wikileaksKey and len(fields) == 4:
            return True
        else:
            return False
    except:
        return False


# load data from file
vin_lines = sc.textFile("/data/bitcoin/vin.csv")
vout_lines_first = sc.textFile("/data/bitcoin/vout.csv")

# filter wikileaks public keys and filter out bad lines
vout_lines = vout_lines_first.filter(is_to_wikileaks)

# filter out bad lines
vin_lines = vin_lines.filter(is_vin_good_line)

# VIN file:
#  0       1      2
# txid, tx_hash, vout

# VOUT File:
#  0      1    2      3
# hash, value, n, publicKey

v_out_dict = vout_lines.map(lambda line: (line.split(',')[0], {}))

vin_lines = vin_lines.map(lambda x: (x.split(',')[0], (x.split(',')[1], x.split(',')[2])))

joined_1 = vin_lines.join(v_out_dict)

joined_1.saveAsTextFile("outBitcoin/TopTen/joined1")

joined_1 = joined_1.map(lambda x: (x[0], x[1][0][0], x[1][0][1]))

joined_1.saveAsTextFile('outBitcoin/TopTen/joined1_mapped')
#
# # Joined 1
# #   0       1      2
# # tx_id, tx_hash, vout

joined_1_dict = joined_1.map(lambda x: ((x[1], x[2]), {}))

vout_lines = vout_lines_first

vout_lines = vout_lines.filter(is_vout_good_line)

vout_lines = vout_lines.map(lambda line: ((line.split(',')[0], line.split(',')[2]), (line.split(',')[1], line.split(',')[3])))

joined_2 = joined_1_dict.join(vout_lines)

#joined2
#    [0][0] [0][1]   [1][0]     [1][1][0]   [1][1][1]
#  (tx_hash, vout), ( {}  ,    (  value   , publicKey))


joined_2.saveAsTextFile('outBitcoin/TopTen/joined2')

joined_2 = joined_2.map(lambda x: (x[1][1][1], float(x[1][1][0])))

unsortedResults = joined_2.reduceByKey(lambda x, y: x+y)

top10 = unsortedResults.takeOrdered(10, lambda x: -x[1])


for x in top10:
    print(x)

top10 = sc.parallelize(top10)

top10.saveAsTextFile("outBitcoin/TopTen/Topten")


