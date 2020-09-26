__kernel void removeShapesInsideShape(__global unsigned char *image, __global unsigned char *result, int size)
{
    unsigned int y = get_global_id(0);
    unsigned int x = get_global_id(1);
    unsigned int sizeX = get_global_size(1);
    unsigned int index = x + y * sizeX;

    if(x > size && sizeX - x > size && y > size && get_global_size(0) - y > size)
    {
        int outer_layer = 0;
        for(int i=(signed)-(size-1)/2; i<=(size-1)/2; i+=size)
        {
            if(outer_layer == 1)
                break;
            for(int j=(signed)-(size-1); j<=(size-1)/2; j++)
            {
                if(image[index+i+j*sizeX] == 255)
                {
                    outer_layer = 1;
                    break;
                }
            }
        }
        for(int i=(signed)-(size-1)/2; i<=(size-1)/2; i+=size)
        {
            if(outer_layer == 1)
                break;
            for(int j=(signed)-(size-1); j<=(size-1)/2; j++)
            {
                if(image[index+j+i*sizeX] == 255)
                {
                    outer_layer = 1;
                    break;
                }
            }
        }

        if(outer_layer == 0)
        {
            for(int i=(signed)-(size-1)/2; i<=(size-1)/2; i++)
            {
                for(int j=(signed)-(size-1); j<=(size-1)/2; j++)
                {
                    result[index+i+j*sizeX] = 70;
                }
            }
        }
        else
        {
            if(result[index] != 70)
                result[index] = image[index];
        }
    }
}

__kernel void removeSmallEdges(__global int *labels, __global int *result, __global int *label, __global int *count, const unsigned int lowThreshold, const unsigned int length)
{
    unsigned int x = get_global_id(1);
    unsigned int y = get_global_id(0);
    unsigned int sizeX = get_global_size(1);
    int index = x + y * sizeX;

    if(labels[index] != 0)
    {
        int position = 1;
        while(position <= length)
        {
            if(label[position-1] == labels[index])
            {
                position--;
                if(count[position] > lowThreshold)
                {
                    result[index] = labels[index];
                }
                else
                    result[index] = 0;
                break;
            }
            position++;
        }
    }
}

__kernel void ConvertLabelsToEdges(__global int *labels, __global unsigned char *edges)
{
    unsigned int x = get_global_id(1);
    unsigned int y = get_global_id(0);
    unsigned int sizeX = get_global_size(1);
    int index = x + y * sizeX;

    if(labels[index] != 0)
        edges[index] = 255;
}