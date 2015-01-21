#for p in {0..2}
#do
#	for q in {0..2}
#	do
p=0
q=0
		for file in res.iter1024.float{*K,[124]M}.d${p}d${q}; do echo $file; awk '/Compute time for q\[1/ {print $(NF-3)}' $file; done
		echo
		for file in res.iter1024.float{*K,[124]M}.d${p}d${q}; do awk '/Marshal time for q\[0/ {print $(NF-3)}' $file; done
		echo
#	done
#done
