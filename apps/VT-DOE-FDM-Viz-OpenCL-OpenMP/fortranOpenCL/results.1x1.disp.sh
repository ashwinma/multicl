# float number comparison
function fcomp() {
    awk -v n1=$1 -v n2=$2 'BEGIN{ if (n1<n2) exit 0; exit 1}'
}

# test and example
function fcomp_test() {
    if fcomp $1 $2; then
       echo "$1<$2"
    else
       echo "$1>=$2"
    fi
}

for file in res.1x1.s1.d[1-2]d[1-2]
do 
	#echo
	#echo $file
	awk '/Dispatch timer:/ {if($(NF-6) > val) val = $(NF-6)} END {print val}' $file
done
exit;
for file in res.1x1.0p12Hz.auto
do 
	#echo
	#echo $file
	awk '/Dispatch timer:/ {if($(NF-6) > val) val = $(NF-6)} END {print val}' $file
done

for file in res.1x1.0p15Hz.auto
do 
	#echo
	#echo $file
	awk '/Dispatch timer:/ {if($(NF-6) > val) val = $(NF-6)} END {print val}' $file
done

for file in res.1x1.0p3Hz.auto
do 
	#echo
	#echo $file
	awk '/Dispatch timer:/ {if($(NF-6) > val) val = $(NF-6)} END {print val}' $file
done

for file in res.1x1.0p6Hz.auto
do 
	#echo
	#echo $file
	awk '/Dispatch timer:/ {if($(NF-6) > val) val = $(NF-6)} END {print val}' $file
done

for file in res.1x1.1p2Hz.d[0-2]d[0-2]
do 
	#echo
	#echo $file
	awk '/Dispatch timer:/ {if($(NF-6) > val) val = $(NF-6)} END {print val}' $file
done

