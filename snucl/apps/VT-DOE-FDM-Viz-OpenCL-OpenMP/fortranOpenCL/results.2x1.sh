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

for file in res.2x1.0p3Hz.d[1-2]d[1-2]
do 
	echo $file
	val0=`awk '/\[0/&&/Device Index 0/ {val+=$(NF-1)} END {print val}' $file`
	val1=`awk '/\[0/&&/Device Index 1/ {val+=$(NF-1)} END {print val}' $file`
	val2=`awk '/\[0/&&/Device Index 2/ {val+=$(NF-1)} END {print val}' $file`
	
	awk '/\[0/&&/Stress Kernel 1/ {val+=$(NF-6)} END {print "Stress\t"val}' $file
	awk '/\[0/&&/Velocity Kernel 1/ {val+=$(NF-6)} END {print "Vel\t"val}' $file
	#echo $val0;
	#echo $val1;
	#echo $val2;
	if [ -z "$val0" ]; then val0=0; fi
	if [ -z "$val1" ]; then val1=0; fi
	if [ -z "$val2" ]; then val2=0; fi
#	fcomp_test $val0 $val1
#	fcomp_test $val0 $val2
#	fcomp_test $val1 $val2
	if fcomp $val0 $val1
	then
		if fcomp $val1 $val2
		then
			echo -e "Raw\t$val2";
		else
			echo -e "Raw\t$val1";
		fi
	else 
		if fcomp $val0 $val2
		then
			echo -e "Raw\t$val2";
		else
			echo -e "Raw\t$val0";
		fi
	fi
done

for file in res.2x1.0p6Hz.d[1-2]d[1-2]
do 
	echo $file
	val0=`awk '/\[0/&&/Device Index 0/ {val+=$(NF-1)} END {print val}' $file`
	val1=`awk '/\[0/&&/Device Index 1/ {val+=$(NF-1)} END {print val}' $file`
	val2=`awk '/\[0/&&/Device Index 2/ {val+=$(NF-1)} END {print val}' $file`
	
	awk '/\[0/&&/Stress Kernel 1/ {val+=$(NF-6)} END {print "Stress\t"val}' $file
	awk '/\[0/&&/Velocity Kernel 1/ {val+=$(NF-6)} END {print "Vel\t"val}' $file
	#echo $val0;
	#echo $val1;
	#echo $val2;
	if [ -z "$val0" ]; then val0=0; fi
	if [ -z "$val1" ]; then val1=0; fi
	if [ -z "$val2" ]; then val2=0; fi
#	fcomp_test $val0 $val1
#	fcomp_test $val0 $val2
#	fcomp_test $val1 $val2
	if fcomp $val0 $val1
	then
		if fcomp $val1 $val2
		then
			echo -e "Raw\t$val2";
		else
			echo -e "Raw\t$val1";
		fi
	else 
		if fcomp $val0 $val2
		then
			echo -e "Raw\t$val2";
		else
			echo -e "Raw\t$val0";
		fi
	fi
done

for file in res.2x1.1p2Hz.d[1-2]d[1-2]
do 
	echo $file
	val0=`awk '/\[0/&&/Device Index 0/ {val+=$(NF-1)} END {print val}' $file`
	val1=`awk '/\[0/&&/Device Index 1/ {val+=$(NF-1)} END {print val}' $file`
	val2=`awk '/\[0/&&/Device Index 2/ {val+=$(NF-1)} END {print val}' $file`
	
	awk '/\[0/&&/Stress Kernel 1/ {val+=$(NF-6)} END {print "Stress\t"val}' $file
	awk '/\[0/&&/Velocity Kernel 1/ {val+=$(NF-6)} END {print "Vel\t"val}' $file
	#echo $val0;
	#echo $val1;
	#echo $val2;
	if [ -z "$val0" ]; then val0=0; fi
	if [ -z "$val1" ]; then val1=0; fi
	if [ -z "$val2" ]; then val2=0; fi
#	fcomp_test $val0 $val1
#	fcomp_test $val0 $val2
#	fcomp_test $val1 $val2
	if fcomp $val0 $val1
	then
		if fcomp $val1 $val2
		then
			echo -e "Raw\t$val2";
		else
			echo -e "Raw\t$val1";
		fi
	else 
		if fcomp $val0 $val2
		then
			echo -e "Raw\t$val2";
		else
			echo -e "Raw\t$val0";
		fi
	fi
done

