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
ThresholdStatusValues statusValues1;
ThresholdStatusValues statusValues2;
ledModeValues ledModeValues;

float th_min = 5.0;
float th_max = 0.0;

byte act_btn;

char on[3] = {'o','n', '\0'};
char off[4] = {'o','f','f', '\0'};
char lo[3] = {'l','o', '\0'};
char hi[3] = {'h','i', '\0'};
char lt[3] = {'l','t', '\0'};
char ht[3] = {'h','t','\0'};


void printD(char str[], float a){
  char aux[2], aux2[4];
  dtostrf( a, 3, 1, aux );
  sprintf(aux2,"%s%s",str,aux);
  MFS.write(aux2);
}

void printBlink(char str[]){
  MFS.write(str);
  MFS.blinkDisplay(DIGIT_ALL, ON);
}

void printBlinkP(char str[], float a){
  Serial.println(str);
  printD(str,a);    
  MFS.blinkDisplay(DIGIT_1, ON);
  MFS.blinkDisplay(DIGIT_2, ON);
}

void printN(char str[]){
  MFS.blinkDisplay(DIGIT_ALL, OFF);
  MFS.write(str);
}

void set_lo_th(){
  Serial.println("lt");
  printD(lo,th_min);
    if ( act_btn == BUTTON_1_PRESSED || act_btn == BUTTON_1_LONG_PRESSED){
      statusValues1 = THRESHOLD_SET;
      MFS.writeLeds(LED_2, ON);

      monitorValues = MONITORING_STOPPED;
    } //ignorar o evento longo

  if(act_btn == BUTTON_2_PRESSED || act_btn == BUTTON_2_LONG_PRESSED){ // mesma coisa no longo e curto
      th_min+=0.1;
  }
  if(act_btn == BUTTON_3_PRESSED || act_btn == BUTTON_3_LONG_PRESSED){// mesma coisa no longo e curto
      th_min-=0.1;
  }
}

void set_hi_th(){
  Serial.println("ht");
  printD(hi,th_max);
    if (act_btn == BUTTON_1_PRESSED || act_btn == BUTTON_1_LONG_PRESSED){
      statusValues2= THRESHOLD_SET;
      MFS.writeLeds(LED_1, ON);
      
      monitorValues = MONITORING_STOPPED;
    } //ignorar o evento longo
    
  
  if(act_btn == BUTTON_2_PRESSED || act_btn == BUTTON_2_LONG_PRESSED){ // mesma coisa no longo e curto
      th_max+=0.1;
  }
    
    
  if(act_btn == BUTTON_3_PRESSED || act_btn == BUTTON_3_LONG_PRESSED){// mesma coisa no longo e curto
      th_max-=0.1;
  }

}

void cycle_led(){
  switch (ledModeValues) {
   case LED_ALL_OFF:
      ledModeValues = LED_1_ON;
      MFS.writeLeds(LED_1, ON);
    break;
    case LED_1_ON:
      MFS.writeLeds(LED_1, OFF);
      ledModeValues = LED_2_ON;
      MFS.writeLeds(LED_2, ON);
    break;
    case LED_2_ON:
      MFS.writeLeds(LED_2, OFF);
      ledModeValues = LED_3_ON;
      MFS.writeLeds(LED_3, ON);
    break;
    case LED_3_ON:
      MFS.writeLeds(LED_3, OFF);
      ledModeValues = LED_4_ON;
      MFS.writeLeds(LED_4, ON);
    break;
    case LED_4_ON:
      MFS.writeLeds(LED_4, OFF);
      ledModeValues = LED_1_ON;
      MFS.writeLeds(LED_1, ON);
    break;
  }
  delay(100);
}

void started(){ //monitor started
  cycle_led();
  int pot = analogRead(POT_PIN);
  float current_volt = pot * VOLTAGE_UNIT;
  
  if(current_volt>th_max){
    printD(hi,current_volt);
    bibi();
  }
  else if(current_volt < th_min){
    printD(lo,current_volt);
    bibi();
  }
  else {
    printBlinkP(on,current_volt); 
  }
  
  if(act_btn == BUTTON_1_PRESSED || act_btn == BUTTON_1_LONG_PRESSED){
    //curto e longo fazem voltar ao estado não configurado
    monitorValues = MONITORING_STOPPED;
    statusValues1 = THRESHOLD_NOT_SET;
    statusValues2 = THRESHOLD_NOT_SET;
    ledModeValues = LED_ALL_OFF;
    // perguntar ao anderson se tem q voltar os th_min e th_max para o valor inicial
  }
    
  
  if(act_btn == BUTTON_2_PRESSED || act_btn == BUTTON_2_LONG_PRESSED){ //curto e longo fazem a mesma coisa
    
      printD(ht,th_max); //somente enquanto estiver pressionado . . .
      delay(500);
  }
    
    
  if(act_btn == BUTTON_3_PRESSED || act_btn == BUTTON_3_LONG_PRESSED){ //curto e longo fazem a mesma coisa
     printD(lt,th_min); //somente enquanto estiver pressionado . . .
      delay(500);
  }
    
}

void stopped(){
  ledModeValues = LED_ALL_OFF;
  printN(off);
  
   if (act_btn == BUTTON_1_SHORT_RELEASE){
      if (statusValues1 == THRESHOLD_SET && statusValues2 == THRESHOLD_SET){
        monitorValues = MONITORING_STARTED;
      }
    }
    
    if (act_btn == BUTTON_2_SHORT_RELEASE){
      printD(ht,5.0);
      delay(500);
    }
    else if (act_btn == BUTTON_2_LONG_RELEASE) { //if long press
      monitorValues = SETTING_HIGH_TH_STARTED;
    }
  
    if ( act_btn == BUTTON_3_SHORT_RELEASE){
      printD(lt,0.0f);
      delay(500);
    }
    else if (act_btn == BUTTON_3_LONG_RELEASE){ //if long press
      monitorValues = SETTING_LOW_TH_STARTED;
      Serial.println("Lowww")
    }
}

byte getCurrentButton(byte btn){
  return btn & B00111111;
}

byte getCurrentAction(byte btn){
  return btn & B11000000;
}

void bibi(){
MFS.beep(4, 4, 3, 3, 50);
}

void setup(){
  Serial.begin(9600);
  Timer1.initialize();
  MFS.initialize(&Timer1);

  monitorValues = MONITORING_STOPPED;
  statusValues1 = THRESHOLD_NOT_SET;
  statusValues2 = THRESHOLD_NOT_SET;
  ledModeValues = LED_ALL_OFF;

  th_min = 0.0f;
  th_max = 5.0f;

}

void loop(){
  act_btn = MFS.getButton();

  switch(monitorValues){
    case MONITORING_STOPPED:
      stopped();
      break;

    case MONITORING_STARTED:
      started();
      break;
      
    case SETTING_HIGH_TH_STARTED:
      set_hi_th();
      break;
    
    case SETTING_LOW_TH_STARTED:
      set_lo_th();
    case default:
      set_lo_th();
      break;
  } 
}MFS.writeLeds(LED_2, ON);