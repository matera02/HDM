/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.mycompany.nsp;
import java.io.IOException;
import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;

/**
 *
 * @author roberto
 */
public class NSPTabuSearch extends NSP {
    
    private int iterations;
    private int tabuTenure;
    private List<int[][][]> tabuList;
    private int[][][] bestSchedule;
    private double bestFitness;
    
    
    public NSPTabuSearch(String filename, int iterations, int tabuTenure) throws IOException{
        super(filename);
        this.iterations = iterations;
        this.tabuTenure = tabuTenure;
        this.tabuList = new ArrayList<>();
    }
    
    private int[][][] copySchedule(int[][][] schedule){
        int[][][] newSchedule = new int[schedule.length][][];
        for(int i = 0; i < schedule.length; i++){
            newSchedule[i] = new int[schedule[i].length][];
            for(int j = 0; j < schedule[i].length; j++){
                newSchedule[i][j] = Arrays.copyOf(schedule[i][j], schedule[i][j].length);
            }
        }
        return newSchedule;
    }
    
    private boolean isTabu(int[][][] schedule){
        for(int[][][] tabuSchedule : tabuList){
            if(Arrays.deepEquals(tabuSchedule, schedule)){
                return true;
            }
        }
        return false;
    }
    
    
    private List<int[][][]> getNeighbors(int[][][] schedule) {
        List<int[][][]> neighbors = new ArrayList<>();
        for(int i = 0; i < super.numNurses; i++){
            for (int k = 0; k < super.numDays; k++){
                for(int s = 0; s < super.numShifts; s++){
                    if (schedule[i][k][s] == 0) {
                        int[][][] newSchedule = copySchedule(schedule);
                        Arrays.fill(newSchedule[i][k], 0);
                        newSchedule[i][k][s] = 1;
                        if (isFeasible(newSchedule) && !isTabu(newSchedule)) {
                            neighbors.add(newSchedule);
                        }
                    }
                }
            }
        }
        return neighbors;
    }
    
    
    public int[][][] getBestSchedule(){
        return bestSchedule;
    }
    
    
    public double getBestFitness(){
        return bestFitness;
    }
    
    
    @Override
    public void run(){
        int[][][] currentSchedule = super.randomSchedule();
        while (!super.isFeasible(currentSchedule)) {
            currentSchedule = super.randomSchedule();
        }
        
        this.bestSchedule = currentSchedule;
        this.bestFitness = super.fitness(currentSchedule);

        for(int iteration = 0; iteration < this.iterations; iteration++) {
            List<int[][][]> neighbors = this.getNeighbors(currentSchedule);
            if(neighbors.isEmpty()){
                break;
            }
            neighbors.sort((n1, n2) -> Double.compare(fitness(n1), fitness(n2)));
            currentSchedule = neighbors.get(0);
            double currentFitness = super.fitness(currentSchedule);

            if(currentFitness < this.bestFitness){
                this.bestSchedule = currentSchedule;
                this.bestFitness = currentFitness;
            }

            this.tabuList.add(currentSchedule);
            if(this.tabuList.size() > this.tabuTenure){
                this.tabuList.remove(0);
            }

            System.out.println("Iteration " + (iteration + 1) + ": Best Fitness = " + this.bestFitness);
        }

        super.printBestSolution(this.bestSchedule, this.bestFitness);
    }
    
    
    public static void main(String[] args) throws IOException{
        long startTime = System.currentTimeMillis();
        String filename = "1.nsp";
        NSPTabuSearch tbs = new NSPTabuSearch(filename, 1000, 5);
        tbs.run();
        long endTime = System.currentTimeMillis();
        double duration = (endTime - startTime) / 1000.0;
        System.out.println("Elapsed time: " + duration + " seconds");
    }
}
