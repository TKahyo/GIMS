#!/bin/sh

#./CONVERTER.sh -f DEMO_DATA1.csv -r 600,899,99 -e 26 -w 27 -l DEMO_LIST.csv -s DEMO_DATA1_conv >log_converter 2>&1 &

function usage {
  cat <<EOM
Usage: $(basename "$0") [OPTION]...
  -h              Here. *necessary
  -f FILE_PATH    Raw MSI csv data (*)
  -d DECIMAL_VALUE    Decimal value of m/z. Round. e.g. 0.01, default=0.01
  -p PLATFORM    MSI platform imscope or solarix, default=imscope
  -r RANGE    Range of m/z (min, max). e.g. 600,899.99 (*)
  -c COLUMN_NUMBER    Start column of m/z. default=3
  -e HEIGHT_OF_PICTURE    Height pixel number of original picture (*)
  -w WIDTH_OF_PICTURE    Width pixel number of original picture (*)
  -l LIST_FILE_PATH    csv file path of m/z list (optional)
  -s SAVE_FILE    Saved file name (*)
EOM
exit 2
}

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
DECIMAL=0.01
PLATFORM="imscope"
COLN=3
STRIDE=1
LIST="None"

while getopts ":f:d:p:r:c:e:w:l:s:h" optKey; do
  case "$optKey" in
    f)
      echo "-f = ${OPTARG}"
      FILE=$OPTARG
      ;;
    d)
      echo "-d = ${OPTARG}"
      DECIMAL=$OPTARG
      ;;
    p)
      echo "-p = ${OPTARG}"
      PLATFORM=$OPTARG
      ;;
    r)
      echo "-r = ${OPTARG}"
      RANGE=$OPTARG
      ;;
    c)
      echo "-c = ${OPTARG}"
      COLN=$OPTARG
      ;;
    e)
      echo "-e = ${OPTARG}"
      HEIGHT=$OPTARG
      ;;
    w)
      echo "-w = ${OPTARG}"
      WIDTH=$OPTARG
      ;;
    l)
      echo "-l = ${OPTARG}"
      LIST=$OPTARG
      ;;
    s)
      echo "-s = ${OPTARG}"
      SAVE=${OPTARG}
      ;;
    '-h'|'--help'|* )
      usage
      ;;
  esac
done

echo "\n"
echo "Script directory : ${SCRIPT_DIR}"
echo "MSI data path    : ${FILE}"
echo "Decimal value    : ${DECIMAL}"
echo "Platform         : ${PLATFORM}"
echo "Range            : ${RANGE}"
echo "Column number    : ${COLN}"
echo "Height of picture: ${HEIGHT}"
echo "Width of picture : ${WIDTH}"
echo "Stride value     : ${STRIDE}"
echo "List file path   : ${LIST}"
echo "Saved file name  : ${SAVE}"
echo "\n"

echo ">> TK_d1_Converting.pl..."
SCRIPT1="$SCRIPT_DIR/scripts/TK_d1_Converting.pl"
CURRENT_DATE=$(date +"%Y-%m-%d_%H-%M-%S")
FILE_NAME="${CURRENT_DATE}.tsv"
perl ${SCRIPT1} --file ${FILE} --unit ${DECIMAL} --machine ${PLATFORM} --save ${FILE_NAME} --range ${RANGE}
ls -lh
echo "\n"

echo ">> TK_d3_Datasetting.py..."
SCRIPT2="$SCRIPT_DIR/scripts/TK_d3_Datasetting.py"
python ${SCRIPT2}  --file ${FILE_NAME} --z 1 --narrow ${LIST} --coln ${COLN} --stride ${STRIDE} --height ${HEIGHT} --width ${WIDTH} --batch 10 --save ${SAVE}
ls -lh
echo "\n"
SAVE_DIR="${SAVE}_dir"
mkdir -p ${SAVE_DIR}
SAVE2="${SAVE}_narrowed.tsv"
mv ${SAVE} ${SAVE_DIR}
mv ${SAVE2} ${SAVE_DIR}
rm ${FILE_NAME}

date
echo ">> Done.\n"


