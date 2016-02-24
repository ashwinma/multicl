
for file in res.1x4.0p15Hz.d[1-2]d[1-2]
do 
	#echo
	#echo $file
	awk '/\[0/&&/Dispatch timer:/ {if($(NF-6) > val) val = $(NF-6)} END {print val}' $file
done


