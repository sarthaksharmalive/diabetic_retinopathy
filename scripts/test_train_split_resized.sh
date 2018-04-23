# select 20% random files from DR and put to validation
ls /home/inspire/Documents/SArthak/dataset/train_resized/DR/ | sort -R | tail -1863 | while read file; do
    # Something involving $file, or you can leave
    # off the while to just get the filenames
    mv /home/inspire/Documents/SArthak/dataset/train_resized/DR/$file /home/inspire/Documents/SArthak/dataset/test_resized/DR/
done
# select 20% random files from NO_DR and put to validation
ls /home/inspire/Documents/SArthak/dataset/train_resized/NO_DR/ | sort -R | tail -5162 | while read file; do
    # Something involving $file, or you can leave
    # off the while to just get the filenames
    mv /home/inspire/Documents/SArthak/dataset/train_resized/NO_DR/$file /home/inspire/Documents/SArthak/dataset/test_resized/NO_DR/
done

