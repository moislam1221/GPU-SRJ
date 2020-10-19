__host__ __device__
double jacobi1DPoisson(const double leftX, const double rightX, const double centerRhs, const double dx)
{
    return (centerRhs*dx*dx + leftX + rightX) / 2.0;
}

__host__ __device__
double jacobi2DPoisson(const double leftX, const double rightX, const double topX, const double bottomX, const double centerRhs, const double dx, const double dy)
{
    // return (centerRhs*dx*dx*dy*dy + dy*dy*(leftX + rightX) + dx*dx*(topX+bottomX)) / (2.0*(dx*dx + dy*dy));
    return (centerRhs*dx*dx + leftX + rightX + topX + bottomX) / 4.0;
    // return (centerRhs*dx*dx*dy*dy + dy*dy*(leftX + rightX) + dx*dx*(topX + bottomX)) / (2.0*(dx*dx + dy*dy));
    // return (2.0*(dx*dx + dy*dy));
    // return (centerRhs*dx*dx + leftX + rightX) / 2.0;
}

__host__ __device__
double jacobi(const double leftMatrix, const double centerMatrix, const double rightMatrix, const double topMatrix, const double bottomMatrix,
             const double leftX, const double centerX, const double rightX, const double topX, const double bottomX, const double centerRhs)
{
    return (centerRhs - (leftMatrix * leftX + rightMatrix * rightX + topMatrix * topX + bottomMatrix * bottomX)) / centerMatrix;
}

__host__ __device__
double jacobi2DPoissonRelaxed(const double leftX, const double centerX, const double rightX, const double topX, const double bottomX, const double centerRhs, const double dx, const double dy, const double omega)
{
    // return (centerRhs*dx*dx*dy*dy + dy*dy*(leftX + rightX) + dx*dx*(topX+bottomX)) / (2.0*(dx*dx + dy*dy));
    return omega * ((centerRhs*dx*dx + leftX + rightX + topX + bottomX) / 4.0) + (1.0 - omega) * centerX;
    // return (centerRhs*dx*dx*dy*dy + dy*dy*(leftX + rightX) + dx*dx*(topX + bottomX)) / (2.0*(dx*dx + dy*dy));
    // return (2.0*(dx*dx + dy*dy));
    // return (centerRhs*dx*dx + leftX + rightX) / 2.0;
}
