#include "city_info.h"

void CITY_INFO::set_info(std::string city, int i, double x1, double y1)
{
	cityname.assign(city);
	index = i;
	x = x1;
	y = y1;
}