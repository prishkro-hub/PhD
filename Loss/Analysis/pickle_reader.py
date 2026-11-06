import pickle

with open(r"C:\Users\royp1\OneDrive - post.bgu.ac.il\Documents\Masters\Yaniv\5umGapRes_6789MHz_data\res6789GHz_5V_full.pickle", "rb") as file:
    data = pickle.load(file)

    print(data)
