# include <stdio.h>
# include <stdlib.h>

# define INT_TYPE 1
# define FLOAT_TYPE 2 

// write the data to the filename; the first value in the file is the array size. 
void write_array2file(void *data, unsigned int type, int size, const char *filename)
{
	int data_unit;
	switch(type)
	{
		case 1:
			data_unit = sizeof(int);
			break;
		case 2:
			data_unit = sizeof(float);
			break;
		default:
			fprintf(stderr, "Writing Error: Undefined data type.\n");
			exit(EXIT_FAILURE);
	}
	
	FILE *fp;
	if((fp = fopen(filename, "wb"))==NULL) 
	{
		fprintf(stderr, "Cannot open file.\n");
		exit(EXIT_FAILURE);
	}
	
	if(fwrite(&size, sizeof(int), 1, fp) != 1)
	{
		fprintf(stderr, "File write error: the size cannot be written.\n");
		exit(EXIT_FAILURE);
	}
	
	if(fwrite(data, data_unit, size, fp) != size)
	{
		fprintf(stderr, "File write error: the data cannot be written.\n");
		exit(EXIT_FAILURE);
	}
	fclose(fp);
}

// get the array size in the filename
int getsize_file2array(const char *filename)
{
	FILE *fp;
	int size;
	if((fp=fopen(filename, "rb"))==NULL) 
	{
		fprintf(stderr, "Cannot open file.\n");
		exit(EXIT_FAILURE);
	}
	
	if(fread(&size, sizeof(int), 1, fp) != 1) 
	{
		fprintf(stderr, "Reading Error: the size cannot be read.\n");
		exit(EXIT_FAILURE);
	}
	fclose(fp);
	return size;
}

// get the array data from the filename and save them to *data pointer.
void read_file2array(void **data, unsigned int type, int size, const char *filename)
{
	int data_unit;
	switch(type)
	{
		case 1:
			data_unit = sizeof(int);
			break;
		case 2:
			data_unit = sizeof(float);
			break;
		default:
			fprintf(stderr, "Reading Error: Undefined data type.\n");
			exit(EXIT_FAILURE);
	}
	
	FILE *fp;
	if((fp=fopen(filename, "rb"))==NULL) 
	{
		fprintf(stderr, "Cannot open file.\n");
		exit(EXIT_FAILURE);
	}
	
	int tmp;
	
	if(fread(&tmp, sizeof(int), 1, fp) != 1) 
	{
		fprintf(stderr, "Reading Error: the size cannot be read.\n");
		exit(EXIT_FAILURE);
	}
	
	if(fread(*data, data_unit, size, fp) != size) 
	{
		if(feof(fp))
		{
			fprintf(stderr, "Premature end of file.");
			exit(EXIT_FAILURE);
		} else
		{
			fprintf(stderr, "File read error.");
			exit(EXIT_FAILURE);
		}
	}
	fclose(fp);

}


