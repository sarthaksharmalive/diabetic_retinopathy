# select 20% random files from DR and put to validation
ls /home/inspire/Documents/SArthak/dataset/train/DR/ | sort -R | tail -1863 | while read file; do
    # Something involving $file, or you can leave
    # off the while to just get the filenames
    mv /home/inspire/Documents/SArthak/dataset/train/DR/$file /home/inspire/Documents/SArthak/dataset/test/DR/
done
# select 20% random files from NO_DR and put to validation
ls /home/inspire/Documents/SArthak/dataset/train/NO_DR/ | sort -R | tail -5162 | while read file; do
    # Something involving $file, or you can leave
    # off the while to just get the filenames
    mv /home/inspire/Documents/SArthak/dataset/train/NO_DR/$file /home/inspire/Documents/SArthak/dataset/test/NO_DR/
done

