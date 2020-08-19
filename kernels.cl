#define PI 3.14159265358979323846

__kernel void GradientCalculation(__global char *image, __global char *result, __global double *angle)
{
    unsigned int sizeX = get_global_size(1);
    unsigned int sizeY = get_global_size(0);

    unsigned int x = get_global_id(1);
    unsigned int y = get_global_id(0);

    float kernelX[3][3] = {{-1, 0, 1},
                           {-3, 0, 3},
                           {-1, 0, 1}};

    float kernelY[3][3] = {{-1, -3, -1},
                           {0, 0, 0},
                           {1, 3, 1}};


    int magX=0,magY=0;

    if(x > 200 && x < sizeX - 200 && y > 30 && y < sizeY - 30)
    {
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
    }
    int index = x + y * sizeX;
    result[index] = sqrt((double)(magX * magX + magY * magY));
    angle[index] = (atan2((double)magY, (double)magX) * 180) / PI;
}


__kernel void NonMaxSuppression(__global char *image, __global double *angle, __global char *result)
{
    unsigned int x = get_global_id(1);
    unsigned int y = get_global_id(0);
    unsigned int sizeX = get_global_size(1);
    int index = x + sizeX * y;

    int q=255, r=255;
    if(0 <= angle[index] < 22.5 || 157.5 <= angle[index] <= 180)
    {
        int neighbor = x + (y + 1) * sizeX;
        q = image[neighbor];
        neighbor = x + (y - 1) * sizeX;
        r = image[neighbor];
    }
    else if(22.5 <= angle[index] < 67.5)
    {
        int neighbor = (x + 1) + (y - 1) * sizeX;
        q = image[neighbor];
        neighbor = (x - 1) + (y + 1) * sizeX;
        r = image[neighbor];
    }
    else if(67.5 <= angle[index] < 112.5)
    {
        int neighbor = (x + 1) + y * sizeX;
        q = image[neighbor];
        neighbor = (x - 1) + y * sizeX;
        r = image[neighbor];
    }
    else if(112.5 <= angle[index] < 157.5)
    {
        int neighbor = (x - 1) + (y - 1) * sizeX;
        q = image[neighbor];
        neighbor = (x + 1) + (y + 1) * sizeX;
        r = image[neighbor];
    }

    if(image[index] >= q && image[index] >=r)
        result[index] = image[index];
    else
        result[index] = 0;
}


__kernel void hysteresis(__global char *image, __global char *result, unsigned int weak, unsigned int strong)
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