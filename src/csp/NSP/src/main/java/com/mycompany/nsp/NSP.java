/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 */

 package com.mycompany.nsp;
 import java.io.BufferedReader;
 import java.io.FileReader;
 import java.io.IOException;
 import java.util.Random;
 
 /**
  *
  * @author roberto
  */
 public class NSP {
     
     private int numNurses;
     private int numDays;
     private int numShifts;
     private int[][] hospitalCoverage;
     private int[][][] nursePreferences;
     protected Random random;
     
     
     protected NSP(String filename) throws IOException{
         NspData data = this.parseNspFile(filename);
         this.numNurses = data.getNumNurses();
         this.numDays = data.getNumDays();
         this.numShifts = data.getNumShifts();
         this.hospitalCoverage = data.getHospitalCoverage();
         this.nursePreferences = data.getNursePreferences();
         this.random = new Random();
     }
     
     
     protected int[][][] randomSchedule() {
         int[][][] schedule = new int[this.getNumNurses()][this.getNumDays()][this.getNumShifts()];
         for(int i = 0; i < this.getNumNurses(); i++){
             for(int k = 0; k < this.getNumDays(); k++){
                 int s = this.random.nextInt(this.getNumShifts());
                 schedule[i][k][s] = 1;
             }
         }
         return schedule;
     }
     
     protected boolean isFeasible(int[][][] schedule) {
         // Controllo tutti i vincoli hard
         // Vincolo 1: Che venga rispettata la copertura minima ospedaliera
         for(int k = 0; k < this.getNumDays(); k++){
             for(int s = 0; s < this.getNumShifts(); s++){
                 int sum = 0;
                 for(int i = 0; i < this.getNumNurses(); i++){
                     sum += schedule[i][k][s];
                 }
                 if(sum < this.getHospitalCoverage()[k][s]){
                     return false;
                 }
             }
         }
         
         
         // Vincolo 2: Ad un turno notturno non può essere seguito un turno di mattina
         for(int i = 0; i < this.getNumNurses(); i++){
             for(int k = 0; k < this.getNumDays() - 1; k++){
                 if(schedule[i][k][2] == 1 && schedule[i][k + 1][0] == 1){
                     return false;
                 }
             }
         }
         
         // Vincolo 3: Esattamente un turno al giorno
         for(int i = 0; i < this.getNumNurses(); i++){
             for(int k = 0; k < this.getNumDays(); k++){
                 int sum = 0;
                 for(int s = 0; s < this.getNumShifts(); s++){
                     sum += schedule[i][k][s];
                 }
                 if (sum != 1) {
                     return false;
                 }
             }
         }
         return true;
     }
     
     
     protected double fitness(int[][][] schedule) {
         
         // Componente che rappresenta l'avversione degli infermieri al turno
         double nurseAversion = 0;
         for(int i = 0; i < this.getNumNurses(); i++){
             for(int k = 0; k < this.getNumDays(); k++){
                 for(int s = 0; s < this.getNumShifts(); s++){
                     nurseAversion += schedule[i][k][s] * this.getNursePreferences()[i][k][s];
                 }
             }
         }
 
         // Componente che rappresenta il costo ospedaliero
         double hospitalCost = 0;
         for(int k = 0; k < this.getNumDays(); k++){
             for(int s = 0; s < this.getNumShifts(); s++){
                 int coverage = 0;
                 for(int i = 0; i < this.getNumNurses(); i++){
                     coverage += schedule[i][k][s];
                 }
                 int diff = coverage - this.getHospitalCoverage()[k][s];
                 hospitalCost += diff;
             }
         }
         
         // Somma pesata
         double lambdaWeight = 0.46;
         return lambdaWeight * nurseAversion + (1 - lambdaWeight) * hospitalCost;
     }
     
     // Da implementare nelle sottoclassi
     protected void run(){};
     
     
     public void printBestSolution(int[][][] bestSchedule, double bestFitness){
         String[] shiftNames = {"Morning", "Afternoon", "Night", "Off"};
         System.out.println("Best Fitness: " + bestFitness);
         System.out.println("Best Schedule:");
         for(int i = 0; i < this.getNumNurses(); i++){
             System.out.println("Nurse " + (i + 1) + ":");
             for(int k = 0; k < this.getNumDays(); k++){
                 int shift = 0;
                 for(int s = 0; s < this.getNumShifts(); s++){
                     if(bestSchedule[i][k][s] == 1){
                         shift = s;
                         break;
                     }
                 }
                 System.out.println("  Day " + (k + 1) + ": " + shiftNames[shift]);
             }
             System.out.println();
         }
     }
     
     
     
     
     
     private NspData parseNspFile(String filename) throws IOException {
         BufferedReader reader = new BufferedReader(new FileReader(filename));
         String line;
         // Lettura delle dimensioni del problema
         line = reader.readLine();
         while(line != null && line.trim().isEmpty()){
             line = reader.readLine();
         }
         if(line == null){
             throw new IOException("Il file è vuoto o mal formattato.");
         }
         String[] firstLine = line.trim().split("\\s+");
         int numNurses = Integer.parseInt(firstLine[0]);
         int numDays = Integer.parseInt(firstLine[1]);
         int numShifts = Integer.parseInt(firstLine[2]);
 
         // Inizializzo le matrici di copertura minima ospedaliera e di preferenza degli infermieri
         int[][] hospitalCoverage = new int[numDays][numShifts];
         int[][][] nursePreferences = new int[numNurses][numDays][numShifts];
 
         // Lettura dei requisiti di copertura
         for(int k = 0; k < numDays; k++){
             line = reader.readLine();
             while(line != null && line.trim().isEmpty()){
                 line = reader.readLine();
             }
             if(line == null){
                 throw new IOException("Numero di giorni nel file non corrisponde ai dati dichiarati.");
             }
             String[] coverageLine = line.trim().split("\\s+");
             for(int j = 0; j < numShifts; j++){
                 hospitalCoverage[k][j] = Integer.parseInt(coverageLine[j]);
             }
         }
 
         // Lettura delle preferenze degli infermieri
         for(int i = 0; i < numNurses; i++){
             line = reader.readLine();
             while(line != null && line.trim().isEmpty()){
                 line = reader.readLine();
             }
             if(line == null){
                 throw new IOException("Numero di infermieri nel file non corrisponde ai dati dichiarati.");
             }
             String[] preferenceLine = line.trim().split("\\s+");
             for(int k = 0; k < numDays; k++){
                 for(int j = 0; j < numShifts; j++){
                     nursePreferences[i][k][j] = Integer.parseInt(preferenceLine[k * numShifts + j]);
                 }
             }
         }
         reader.close();
         return new NspData(numNurses, numDays, numShifts, hospitalCoverage, nursePreferences);
     }
 
     /*public static void main(String[] args) {
         System.out.println("Hello World!");
     }*/
 
     /**
      * @return the numNurses
      */
     public int getNumNurses() {
         return numNurses;
     }
 
     /**
      * @return the numDays
      */
     public int getNumDays() {
         return numDays;
     }
 
     /**
      * @return the numShifts
      */
     public int getNumShifts() {
         return numShifts;
     }
 
     /**
      * @return the hospitalCoverage
      */
     public int[][] getHospitalCoverage() {
         return hospitalCoverage;
     }
 
     /**
      * @return the nursePreferences
      */
     public int[][][] getNursePreferences() {
         return nursePreferences;
     }
 }
 