import csv
import chardet
import os 

# # To create positive.csv
# input_file = r"Data\rt-polaritydata\rt-polarity.pos"
# output_file = 'positive.csv'

# To create negative.csv
input_file = r"Data\rt-polaritydata\rt-polarity.neg"
output_file = 'negative.csv'

try:
    with open(input_file, 'rb') as f:
        raw_data = f.read()
        enc = chardet.detect(raw_data)
        print(enc)
        encd = enc['encoding']

    with open(input_file, 'r', encoding = encd) as file1:
        data = file1.readlines() 

    with open(output_file, 'w', newline='', encoding = encd) as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['text']) 

        for index, line in enumerate(data):
            stripped_line = line.strip()   
            if stripped_line.endswith(''):  
                csv_writer.writerow([str(stripped_line)])
            else:
                print(f"{index+1} : {stripped_line}\n")

    print(f"Valid samples have been written to '{output_file}'")

except UnicodeEncodeError as e:
    print(f"Encoding error : {e}")
except FileNotFoundError:
    print(f"Error: File '{input_file}' not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
