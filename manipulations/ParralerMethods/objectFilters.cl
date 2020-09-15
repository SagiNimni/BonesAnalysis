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