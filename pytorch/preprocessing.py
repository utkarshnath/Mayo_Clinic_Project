import csv
import os
import shutil

def get_patiest_ids(path):
    key_table_path = '/data/yyang409/unath/Y90-Project/patientKey/key_table.csv'
    with open(key_table_path, newline='') as csvfile:
         spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
         p_map = {}
         for index, row in enumerate(spamreader):
             p_map[row[2]] = row[1]

    with open(path, newline='') as csvfile:
         spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
         mymap = {}
         for index, row in enumerate(spamreader):
             p_id = row[1].split('_')[0]
             if p_id not in mymap:
                if p_id not in p_map:
                   mymap[p_id] = p_map[p_id.split('-')[0]]
                else:
                   mymap[p_id] = p_map[p_id]
    
    p_lsit = []
    for k in mymap.keys():
        p_lsit.append(k)
    return p_lsit

def check_patient_presence(path):
    key_table = "/scratch/unath/Y90-Project/patientKey/key_table.csv"
    myset = {''}

    with open(path, 'r') as read_obj:
             csv_reader = csv.reader(read_obj)
             item_list = list(csv_reader)
 
    for item in item_list:
        myset.add(item[0])

    with open(key_table, newline='') as csvfile:
         spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
         for index, row in enumerate(spamreader):
             if index==0 or int(row[0])>53:
                 continue
             #print(row[2])
             if row[2] not in myset:
                print(row[2])


check_patient_presence("/scratch/unath/Y90-Project/train/train_patient_ids.csv")
