if [ $# -ne 2 ]
then
	echo "Usage: <NumRepeat> <output.cubin>"
	exit 0
fi

gcc -D REPEAT=repeat$1 -E -x c -C  -P asm |sed  's/---/\n/g' > asm-final ;
gcc -E -x c -C  -P stub-template > stub
gcc -E -x c -C  -P asm-final > asm-final-tmp
mv asm-final-tmp asm-final
asfermi.exe asm-final -o $2
