if [ $# -ne 2 ]
then
	echo "Usage: <ASM file> <NumRepeat>"
	exit 0
fi


gcc -D REPEAT=repeat$2 -E -x c -C  -P $1 |sed  's/---/\n/g' > ${1}-out ;
gcc -E -x c -C  -P stub-manual > stub
gcc -E -x c -C  -P ${1}-out > ${1}-out-tmp
mv ${1}-out-tmp ${1}-out
asfermi.exe ${1}-out -o $1.cubin
