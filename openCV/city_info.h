#include <string>

class CITY_INFO{
public:
	double x;
    double y;
    std::string cityname;
	int index;
	CITY_INFO(){}
	void set_info(std::string, int , double, double );
};