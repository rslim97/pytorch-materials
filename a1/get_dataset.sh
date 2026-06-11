# pwd
# Get directory of the script
DIR="$( cd "$( dirname "$0" )" && pwd )"
echo $DIR
DATA_ROOT=$DIR/dataset 
mkdir -p $DATA_ROOT
chmod 600 ~/.kaggle/kaggle.json
# Sanity check if able to access kaggle 
kaggle datasets list
kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images -p "$DATA_ROOT" --unzip