# imlist is the file containing images of the current directory, you want to check their labels from trainLabels.csv
# imlist can be generated from command line: ls *.jpeg | awk -F'.' '{print $1}' > imlist
for id in `cat imlist`; do grep ^$id, trainLabels.csv; done | awk -F',' '{t=$1;$1=$2;$2=t;print $1"_"$2}'
# it will give output like this: <image_label>_<file_name>
# eg. for 16_left output is 4_16_left
