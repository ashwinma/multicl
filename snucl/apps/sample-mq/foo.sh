for p in {1..2}
do
	for q in {1..2}
	do
		for file in res.iter1024.float4M.d${p}d${q}; do awk '/Compute time for Q\[1/ {print $(NF-3)}' $file; done
	done
done
echo 
for p in {1..2}
do
	for q in {1..2}
	do
		for file in res.iter1024.float4M.d${p}d${q}; do awk '/Marshal time for Q\[0/ {print $(NF-3)}' $file; done
	done
done
echo
for p in {1..2}
do
	for q in {1..2}
	do
		for file in res.iter1024.float2M.d${p}d${q}; do awk '/Compute time for Q\[1/ {print $(NF-3)}' $file; done
	done
done
echo 
for p in {1..2}
do
	for q in {1..2}
	do
		for file in res.iter1024.float2M.d${p}d${q}; do awk '/Marshal time for Q\[0/ {print $(NF-3)}' $file; done
	done
done
for p in {1..2}
do
	for q in {1..2}
	do
		for file in res.iter1024.float1M.d${p}d${q}; do awk '/Compute time for Q\[1/ {print $(NF-3)}' $file; done
	done
done
echo 
for p in {1..2}
do
	for q in {1..2}
	do
		for file in res.iter1024.float1M.d${p}d${q}; do awk '/Marshal time for Q\[0/ {print $(NF-3)}' $file; done
	done
done
echo
for p in {1..2}
do
	for q in {1..2}
	do
		for file in res.iter1024.float512K.d${p}d${q}; do awk '/Compute time for Q\[1/ {print $(NF-3)}' $file; done
	done
done
echo 
for p in {1..2}
do
	for q in {1..2}
	do
		for file in res.iter1024.float512K.d${p}d${q}; do awk '/Marshal time for Q\[0/ {print $(NF-3)}' $file; done
	done
done
echo
for p in {1..2}
do
	for q in {1..2}
	do
		for file in res.iter1024.float256K.d${p}d${q}; do awk '/Compute time for Q\[1/ {print $(NF-3)}' $file; done
	done
done
echo 
for p in {1..2}
do
	for q in {1..2}
	do
		for file in res.iter1024.float256K.d${p}d${q}; do awk '/Marshal time for Q\[0/ {print $(NF-3)}' $file; done
	done
done
