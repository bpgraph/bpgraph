#########################################################################
# File Name: eliminateProb.sh
# Author: ZhangHeng
# mail: zhanghenglab@gmail.com


awk '{$3="";print $0}'  filename > newfile
