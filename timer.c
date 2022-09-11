#include "timer.h"
/*
* float timedifference_msec(struct timeval t0, struct timeval t1)
*
* Recebe uma marca de tempo t0 e outra marca de tempo t1 (ambas do
* tipo struct timeval) e retorna a diferenca de tempo (delta) entre
* t1 e t0 em milisegundos (tipo float).
*/
float timedifference_msec(struct timeval t0, struct timeval t1)
{
return (t1.tv_sec - t0.tv_sec) * 1000.0f + (t1.tv_usec - t0.tv_usec) / 1000.0f;
}