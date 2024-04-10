#pragma once

#if !defined(__CUDACC__)
    #define CUDA_CALLABLE
    #define CUDA_CALLABLE_DEVICE
#else
    #define CUDA_CALLABLE __host__ __device__ 
    #define CUDA_CALLABLE_DEVICE __device__
#endif

// our amazing Wraparound IMage lib
namespace wim
{

struct Color
{
    float r, g, b;

    CUDA_CALLABLE Color(float r = 0.0f, float g = 0.0f, float b = 0.0f)
        : r(r), g(g), b(b)    
    {
    }
};

struct Coord
{
    int x, y;

    CUDA_CALLABLE Coord(int x = 0, int y = 0)
        : x(x), y(y)
    {
    }
};

class Image
{
    int mWidth;
    int mHeight;
    Color* mData;

public:
    CUDA_CALLABLE Image()
        : mWidth(0), mHeight(0), mData(nullptr)
    {
    }

    CUDA_CALLABLE Image(int width, int height, Color* data)
        : mWidth(width), mHeight(height), mData(data)
    {
    }

    CUDA_CALLABLE int getWidth() const
    {
        return mWidth;
    }

    CUDA_CALLABLE int getHeight() const
    {
        return mHeight;
    }

    CUDA_CALLABLE const Color* getData() const
    {
        return mData;
    }

    CUDA_CALLABLE Color* getData()
    {
        return mData;
    }

    CUDA_CALLABLE Coord wrapCoord(const Coord& coord) const
    {
        int x = coord.x;
        int y = coord.y;

        while (x < 0)
            x += mWidth;
        while (x >= mWidth)
            x -= mWidth;

        while (y < 0)
            y += mHeight;
        while (y >= mHeight)
            y -= mHeight;
        
        return Coord(x, y);
    }

    CUDA_CALLABLE Color getPixel(const Coord& coord) const
    {
        if (mData)
        {
            Coord wc = wrapCoord(coord);
            return mData[wc.y * mWidth + wc.x];
        }
        return Color(1.0f, 0.0f, 1.0f);
    }

    CUDA_CALLABLE void setPixel(const Coord& coord, const Color& color)
    {
        if (mData)
        {
            Coord wc = wrapCoord(coord);
            mData[wc.y * mWidth + wc.x] = color;
        }
    }
};

} // end of namespace wim
