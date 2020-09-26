#define PI 3.14159265358979323846

__kernel void GradientCalculation(__global unsigned char *image, __global unsigned char *result, __global double *angle, __global unsigned char *mask, unsigned int gradientRatio, float a, float b, float r)
{
    unsigned int sizeX = get_global_size(1);
    unsigned int sizeY = get_global_size(0);

    unsigned int x = get_global_id(1);
    unsigned int y = get_global_id(0);
    int i = x + y * sizeX;

    if(mask[i] == 0)
    {
        float distance = sqrt((float)((x-a)*(x-a) + (y-b)*(y-b)));
        if(distance <= r)
        {
            float kernelX[3][3] = {{-1, 0, 1},
                                   {(signed)(-gradientRatio), 0, gradientRatio},
                                   {-1, 0, 1}};

            float kernelY[3][3] = {{-1, (signed)(-gradientRatio), -1},
                                   {0, 0, 0},
                                   {1, gradientRatio, 1}};


            int magX=0,magY=0;
            for(int a=0; a<3; a++)
            {
                for(int b=0; b<3; b++)
                {
                    int xn = x + a - 1;
                    int yn = y + b - 1;

                    int index = xn + yn * sizeX;
                    magX += image[index] * kernelX[a][b];
                    magY += image[index] * kernelY[a][b];
                }
            }
            int index = x + y * sizeX;
            result[index] = sqrt((double)(magX * magX + magY * magY));
            angle[index] = (atan2((double)magY, (double)magX) * 180) / PI;
        }
    }
    else
        result[i] = 0;
}


__kernel void NonMaxSuppression(__global unsigned char *image, __global double *angle, __global unsigned char *result)
{
    unsigned int x = get_global_id(1);
    unsigned int y = get_global_id(0);
    unsigned int sizeX = get_global_size(1);
    int index = x + sizeX * y;

    int q=255, r=255;
    if(((0 <= angle[index]) && (angle[index] < 22.5)) || ((157.5 <= angle[index]) && (angle[index] <= 180)))
    {
        int neighbor = x + (y + 1) * sizeX;
        q = image[neighbor];
        neighbor = x + (y - 1) * sizeX;
        r = image[neighbor];
    }
    else if((22.5 <= angle[index]) && (angle[index] < 67.5))
    {
        int neighbor = (x + 1) + (y - 1) * sizeX;
        q = image[neighbor];
        neighbor = (x - 1) + (y + 1) * sizeX;
        r = image[neighbor];

    }
    else if((67.5 <= angle[index]) && (angle[index] < 112.5))
    {
        int neighbor = (x + 1) + y * sizeX;
        q = image[neighbor];
        neighbor = (x - 1) + y * sizeX;
        r = image[neighbor];
    }
    else if((112.5 <= angle[index]) && (angle[index] < 157.5))
    {
        int neighbor = (x - 1) + (y - 1) * sizeX;
        q = image[neighbor];
        neighbor = (x + 1) + (y + 1) * sizeX;
        r = image[neighbor];
    }

    if((image[index] >= q) && (image[index] >=r))
        result[index] = image[index];
    else
        result[index] = 0;
}


__kernel void hysteresis(__global unsigned char *image, __global unsigned char *result, unsigned int weak, unsigned int strong)
{
    unsigned int x = get_global_id(1);
    unsigned int y = get_global_id(0);
    unsigned int sizeX = get_global_size(1);
    int index = x + sizeX * y;

    if(image[index] == weak)
    {
        if(image[index+1-sizeX] == strong || image[index+1] == strong || image[index+1+sizeX] == strong || image[index - sizeX] == strong || image[index+sizeX] == strong || image[index-1-sizeX] == strong || image[index-1] == strong ||image[index-1+sizeX])
            result[index] = strong;
        else
            result[index] = 0;
    }
    else
    {
        result[index] = image[index];
    }
}