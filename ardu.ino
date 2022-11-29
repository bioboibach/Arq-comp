/* Programa de teste do Display do Shield Multi-função
  Blog Eletrogate - https://blog.eletrogate.com/guia-completo-do-shield-multi-funcoes-para-arduino
  Arduino UNO - IDE 1.8.5 - Shield Multi-função para Arduino
  Gustavo Murta   13/junho/2018
  Baseado em http://www.cohesivecomputing.co.uk/hackatronics/arduino-multi-function-shield/part-1/
*/
 
#include <TimerOne.h>                     // Bibliotec TimerOne 
#include <MultiFuncShield.h>              // Biblioteca Multifunction shield
#include <Wire.h>

#define HIGH_VOLTAGE_LIMIT 5.0
#define LOW_VOLTAGE_LIMIT 0.0
#define VOLTAGE_UNIT 0.0049


enum MonitorModeValues{
  MONITORING_STOPPED,
  MONITORING_STARTED,
  SETTING_HIGH_TH_STARTED,
  SETTING_LOW_TH_STARTED
};

enum ThresholdStatusValues{
 THRESHOLD_NOT_SET,
 THRESHOLD_SET
};


enum ledModeValues{
  LED_ALL_OFF,
  LED_1_ON,
  LED_2_ON,
  LED_3_ON,
  LED_4_ON
};

MonitorModeValues monitorValues;
ThresholdStatusValues statusValues;
ledModeValues ledMode;

float th_min;
float th_max;

byte act_btn;
byte num_btn;

void printd(char[] str){
 MFS.write(str);
}


void printBlink(char[] str){
  MFS.write(str);
  MFS.blinkDisplay(DIGIT_ALL, ON);
}

void printBlinkP(char[] str, float a){
  char aux[2], aux2[4];
  dtostrf( a, 3, 1, aux );
  sprintf(aux2,"%s%s",str,aux);
  Serial.println(aux2);
  MFS.write(aux2);    
  MFS.blinkDisplay(DIGIT_1, ON);
  MFS.blinkDisplay(DIGIT_2, ON);
}

void low_th(){
  printd("lo");
  if(num_btn == 1)
    if (act_btn == BUTTON_PRESSED_IND || act_btn == BUTTON_SHORT_RELEASE_IND){

    }
    else { //if long press

    }
  
  if(num_btn == 2)
    if (act_btn == BUTTON_PRESSED_IND || act_btn == BUTTON_SHORT_RELEASE_IND){

    }
    else{ //if long press

    }
  if(num_btn == 3)
    if (act_btn == BUTTON_PRESSED_IND || act_btn == BUTTON_SHORT_RELEASE_IND){

    }
    else{ //if long press

    }
}

void hi_th(){
  printBlink("hi");
  if(num_btn == 1)
    if (act_btn == BUTTON_PRESSED_IND || act_btn == BUTTON_SHORT_RELEASE_IND){

    }
    else { //if long press

    }
  
  if(num_btn == 2)
    if (act_btn == BUTTON_PRESSED_IND || act_btn == BUTTON_SHORT_RELEASE_IND){

    }
    else{ //if long press

    }
  if(num_btn == 3)
    if (act_btn == BUTTON_PRESSED_IND || act_btn == BUTTON_SHORT_RELEASE_IND){

    }
    else{ //if long press

    }
}

void started(){ //monitor started
  printBlink("on");
  if(num_btn == 1)
    if (act_btn == BUTTON_PRESSED_IND || act_btn == BUTTON_SHORT_RELEASE_IND){
      
    }
    else { //if long press

    }
  
  if(num_btn == 2)
    if (act_btn == BUTTON_PRESSED_IND || act_btn == BUTTON_SHORT_RELEASE_IND){

    }
    else{ //if long press

    }
  if(num_btn == 3)
    if (act_btn == BUTTON_PRESSED_IND || act_btn == BUTTON_SHORT_RELEASE_IND){

    }
    else{ //if long press

    }
}

void stopped(){
  ledMode = LED_ALL_OFF;
  MFS.write("off");
  
  if(num_btn == 1) 
    if (act_btn == BUTTON_PRESSED_IND || act_btn == BUTTON_SHORT_RELEASE_IND){
      if (statusValues == THRESHOLD_SET){
        //start monitor
      }
    }
  
  if(num_btn == 2)
    if (act_btn == BUTTON_PRESSED_IND || act_btn == BUTTON_SHORT_RELEASE_IND){
      
    }
    else{ //if long press

    }
  if(num_btn == 3)
    if (act_btn == BUTTON_PRESSED_IND || act_btn == BUTTON_SHORT_RELEASE_IND){

    }
    else{ //if long press

    }
}

byte getCurrentButton(byte btn){
  return btn & B00111111;
}

byte getCurrentAction(byte btn){
  return btn & B11000000;
}



void setup(){
  Serial.begin(9600);
  Timer1.initialize();
  MFS.initialize(&Timer1);

  monitorValues = MONITORING_STOPPED;
  statusValues = THRESHOLD_NOT_SET
  ledMode = LED_ALL_OFF;

  th_min = LOW_VOLTAGE_LIMIT;
  th_max = HIGH_VOLTAGE_LIMIT;

}

void loop(){
  
  byte btn = MFS.getButton();
  num_btn = getCurrentButton(btn);
  act_btn = getCurrentAction(btn);

  switch(monitorValues){
    case MONITORING_STOPPED:
      stopped();
      
      break;

    case MONITORING_STARTED:
      started();
      break;
      
    case SETTING_HIGH_TH_STARTED:
      hi_th();
      break;
    
    case SETTING_LOW_TH_STARTED:
      lo_th();
      break;
  }
}


void ledBlink(){ // exemplo 
  switch (countDownMode)
 {
  case COUNTING_STOPPED:
        if (btn == BUTTON_1_SHORT_RELEASE && (minutes + seconds) > 0)
        {
          // inicia o temporizador
          countDownMode = COUNTING_STARTED;
          ledModeValue = LED_ALL_OFF;
          MFS.writeLeds(LED_ALL, OFF);
        }
        else if (btn == BUTTON_1_LONG_PRESSED)
        {
          // reset do temporizador
          tenths = 0;
          seconds = 0;
          minutes = 0;
          MFS.write(minutes*100 + seconds);
          MFS.blinkDisplay(DIGIT_ALL, OFF);
          ledModeValue = LED_ALL_OFF;
          MFS.writeLeds(LED_ALL, OFF);
        }
        else if (btn == BUTTON_2_PRESSED || btn == BUTTON_2_LONG_PRESSED)
        {
          // ajuste dos minutos
          minutes++;
          if (minutes > 60)
          {
            minutes = 0;
          }
          MFS.write(minutes*100 + seconds);
        }
        else if (btn == BUTTON_3_PRESSED || btn == BUTTON_3_LONG_PRESSED)
        {
          // ajuste dos segundos
          seconds += 10;
          if (seconds >= 60)
          {
            seconds = 0;
          }
          MFS.write(minutes*100 + seconds);
        }
  break;

 case COUNTING_STARTED:
        if (btn == BUTTON_1_SHORT_RELEASE || btn == BUTTON_1_LONG_RELEASE)
        {
          // interrompe o temporizador
          countDownMode = COUNTING_STOPPED;
          ledModeValue = LED_ALL_OFF;
          MFS.writeLeds(LED_ALL, OFF);
        }
        else
        {
          // continua a contagem regressiva
          tenths++;

          if (tenths == 10)
          {
            tenths = 0;
            seconds--;

            if (seconds < 0 && minutes > 0)
            {
              seconds = 59;
              minutes--;
            }

            if (minutes == 0 && seconds == 0)
            {
              // temporizador atingiu ozero, então toca o alarme e imprime End
              countDownMode = COUNTING_STOPPED;
              MFS.write("End");                   // imprime End no display
              MFS.blinkDisplay(DIGIT_ALL, ON);    // pisca o display
             // MFS.beep(50, 50, 3);                // toca o beep 3 vezes, 500 milisegundos on / 500 off
 
              ledModeValue = LED_ALL_OFF;
              MFS.writeLeds(LED_ALL, OFF);        // apaga os LEDs
            }

          if (countDownMode != COUNTING_STOPPED) {
            MFS.write(minutes*100 + seconds);
            switch (ledModeValue)
            {
             case LED_ALL_OFF:
                ledModeValue = LED_1_ON;
                MFS.writeLeds(LED_1, ON);
              break;
              case LED_1_ON:
                MFS.writeLeds(LED_1, OFF);
                ledModeValue = LED_2_ON;
                MFS.writeLeds(LED_2, ON);
              break;
              case LED_2_ON:
                MFS.writeLeds(LED_2, OFF);
                ledModeValue = LED_3_ON;
                MFS.writeLeds(LED_3, ON);
              break;
              case LED_3_ON:
                MFS.writeLeds(LED_3, OFF);
                ledModeValue = LED_4_ON;
                MFS.writeLeds(LED_4, ON);
              break;
              case LED_4_ON:
                MFS.writeLeds(LED_4, OFF);
                ledModeValue = LED_1_ON;
                MFS.writeLeds(LED_1, ON);
              break;
            }
          }
        }
        delay(100);
      }
  break;
 }
}
