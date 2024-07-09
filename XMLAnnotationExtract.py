import xml.etree.ElementTree as ET
import csv

# Define the input XML file and output CSV file
input_xml_file = 'annotations2.xml'
output_csv_file = 'annotations2.csv'

tree = ET.parse(input_xml_file)
root = tree.getroot()

# Open the CSV file for writing
with open(output_csv_file, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    
    # Write the header row
    writer.writerow(['image_name', 'xmin', 'xmax', 'ymin', 'ymax','label'])
    
    # Iterate over each image element
    for image in root.findall('image'):
        image_name = image.get('name')
        
        # Iterate over each box element within the image
        for box in image.findall('box'):
            xmin = box.get('xtl')
            xmax = box.get('xbr')
            ymin = box.get('ytl')
            ymax = box.get('ybr')
            label_name = box.get('label')
            # Write the row to the CSV file
            writer.writerow([image_name, xmin, ymin, xmax, ymax,label_name])

print(f"Annotations successfully exported to {output_csv_file}")
