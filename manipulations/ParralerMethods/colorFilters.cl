__kernel void makeBackground(__global unsigned char (*image)[3], __global unsigned char (*result)[3], unsigned char lowMaskRed, unsigned char lowMaskGreen, unsigned char lowMaskBlue, unsigned char highMaskRed, unsigned char highMaskGreen, unsigned char highMaskBlue)
{
    unsigned int x = get_global_id(1);
    unsigned int y = get_global_id(0);
    unsigned int sizeX = get_global_size(1);
    int index = x + y * sizeX;

    char red = image[index][0];
    char green = image[index][1];
    char blue = image[index][2];

    if(red >= lowMaskRed && green >= lowMaskGreen && blue >= lowMaskBlue && red <= highMaskRed && green <= highMaskGreen && blue <= highMaskBlue)
    {
        result[index][0] = 255;
        result[index][1] = 255;
        result[index][2] = 255;
    }
    else
    {
        result[index][0] = 0;
        result[index][1] = 0;
        result[index][2] = 0;
    }
}