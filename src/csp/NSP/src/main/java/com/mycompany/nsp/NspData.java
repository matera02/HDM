/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.mycompany.nsp;

/**
 *
 * @author roberto
 */
public class NspData {
    
    private int numNurses;
    private int numDays;
    private int numShifts;
    private int[][] hospitalCoverage;
    private int[][][] nursePreferences;
            
    public NspData(int numNurses, int numDays, int numShifts, int[][] hospitalCoverage, int[][][] nursePreferences){
        this.numNurses = numNurses;
        this.numDays = numDays;
        this.numShifts = numShifts;
        this.hospitalCoverage = hospitalCoverage;
        this.nursePreferences = nursePreferences;
    }
    
    public int getNumNurses() {
        return numNurses;
    }
    
    public int getNumDays() {
        return numDays;
    }
    
    public int getNumShifts() {
        return numShifts;
    }
    
    public int[][] getHospitalCoverage() {
        return hospitalCoverage;
    }
    
    public int[][][] getNursePreferences() {
        return nursePreferences;
    }
}
