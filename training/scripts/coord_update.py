

filenum = 0 # whichever image xml file number you want to start at

lastnum = 1000 # last file you want to update

# whatever coordinates you want to fill in the xml files
coord = """
        <object>
            <name>flag</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>80</xmin>
                <ymin>55</ymin>
                <xmax>116</xmax>
                <ymax>91</ymax>
            </bndbox>
        </object>
        <object>
            <name>flag</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>86</xmin>
                <ymin>390</ymin>
                <xmax>125</xmax>
                <ymax>429</ymax>
            </bndbox>
        </object>
        <object>
            <name>flag</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>463</xmin>
                <ymin>401</ymin>
                <xmax>502</xmax>
                <ymax>440</ymax>
            </bndbox>
        </object>
        <object>
            <name>flag</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>474</xmin>
                <ymin>123</ymin>
                <xmax>516</xmax>
                <ymax>165</ymax>
            </bndbox>
        </object>
    """

while filenum <= lastnum:   # until all files have been update

    # creates modified copy of xml file in a "post_update" folder you must create first. This program is designed such that this 
    # script is in the same directory as the images and xml files.
    with open(f"img_{filenum}.xml", "r") as prev_file, open(f"post_update/img_{filenum}.xml", "w") as new_file:    
        prev_contents = prev_file.readlines()
        # Now prev_contents is a list of strings and you may add the new line to this list at any position
        prev_contents.insert(len(prev_contents)-1, coord) # This will add the above text right below the closing "</annotation>" tag
        new_file.write("".join(prev_contents))

    filenum += 1