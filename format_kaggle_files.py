import numpy as np
import datatable as dt
from datetime import datetime
from datatable import f,join,sort
import sys
import os
import json

n = len(sys.argv)

if n == 1:
    print("No input path")
    sys.exit()
    
inPath = sys.argv[1]
outDir = os.path.dirname(inPath)
outPath = outDir + "/formatted_transactions.csv"

raw = dt.fread(inPath, columns = dt.str32)

currency = dict()
paymentFormat = dict()
bankAcc = dict()
account = dict()

def get_dict_val(name, collection):
    if name in collection:
        val = collection[name]
    else:
        val = len(collection)
        collection[name] = val
    return val

header = "EdgeID,from_id,to_id,Timestamp,\
Amount Sent,Sent Currency,Amount Received,Received Currency,\
Payment Format,Is Laundering\n"

firstTs = -1

with open(outPath, 'w') as writer:
    writer.write(header)
    for i in range(raw.nrows):
        datetime_object = datetime.strptime(raw[i,"Timestamp"], '%Y/%m/%d %H:%M')
        ts = datetime_object.timestamp()
        day = datetime_object.day
        month = datetime_object.month
        year = datetime_object.year
        hour = datetime_object.hour
        minute = datetime_object.minute

        if firstTs == -1:
            startTime = datetime(year, month, day)
            firstTs = startTime.timestamp() - 10

        ts = ts - firstTs

        cur1 = get_dict_val(raw[i,"Receiving Currency"], currency)
        cur2 = get_dict_val(raw[i,"Payment Currency"], currency)

        fmt = get_dict_val(raw[i,"Payment Format"], paymentFormat)

        fromAccIdStr = raw[i,"From Bank"] + raw[i,2]
        fromId = get_dict_val(fromAccIdStr, account)

        toAccIdStr = raw[i,"To Bank"] + raw[i,4]
        toId = get_dict_val(toAccIdStr, account)

        amountReceivedOrig = float(raw[i,"Amount Received"])
        amountPaidOrig = float(raw[i,"Amount Paid"])

        isl = int(raw[i,"Is Laundering"])

        line = '%d,%d,%d,%d,%f,%d,%f,%d,%d,%d\n' % \
                    (i,fromId,toId,ts,amountPaidOrig,cur2, amountReceivedOrig,cur1,fmt,isl)

        writer.write(line)

# Persist the encoding dictionaries + firstTs alongside formatted_transactions.csv
# so downstream tools (llm_reasoning.py / mine_fewshot_candidates.py) can turn
# the integer-encoded from_id/to_id/currency/payment-format columns and the
# firstTs-relative Timestamp column back into human-readable values (real
# currency names, real account identifiers, real calendar timestamps) for LLM serialization
 
with open(outDir + "/currency_map.json", "w") as f_out:
    json.dump({v: k for k, v in currency.items()}, f_out)
 
with open(outDir + "/payment_format_map.json", "w") as f_out:
    json.dump({v: k for k, v in paymentFormat.items()}, f_out)
 
with open(outDir + "/account_map.json", "w") as f_out:
    json.dump({v: k for k, v in account.items()}, f_out)
 
# firstTs lets downstream code reconstruct real calendar timestamps from the
# firstTs-relative "Timestamp" column: absolute_epoch_seconds = firstTs + row["Timestamp"]
with open(outDir + "/format_meta.json", "w") as f_out:
    json.dump({
        "firstTs": firstTs,
        "source_file": os.path.basename(inPath),
        "n_transactions": raw.nrows,
    }, f_out)
 
print(f"Wrote currency_map.json, payment_format_map.json, account_map.json, "
      f"format_meta.json to {outDir}")

formatted = dt.fread(outPath)
formatted = formatted[:,:,sort(3)]

formatted.to_csv(outPath)