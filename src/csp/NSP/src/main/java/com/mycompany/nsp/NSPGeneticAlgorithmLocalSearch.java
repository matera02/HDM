/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.mycompany.nsp;

import java.io.IOException;
import java.util.Arrays;

/**
 *
 * @author roberto
 */
public class NSPGeneticAlgorithmLocalSearch extends NSPGeneticAlgorithm {
    
    
    private int localSearchIterations;
    
    public NSPGeneticAlgorithmLocalSearch(String filename, int populationSize, 
            int generations, double mutationRate, 
            int localSearchIterations, int crossoverType) throws IOException{
        super(filename, populationSize, generations, mutationRate, crossoverType);
        this.localSearchIterations = localSearchIterations;
    }
    
    @Override
    protected void mutate(int[][][] schedule) {
        if (random.nextDouble() < super.mutationRate) {
            int i = random.nextInt(super.numNurses);
            int k = random.nextInt(super.numDays);
            int s = random.nextInt(super.numShifts);
            Arrays.fill(schedule[i][k], 0);
            schedule[i][k][s] = 1;
        }
        int[][][] newSchedule = this.localSearch(schedule);
        System.arraycopy(newSchedule, 0, schedule, 0, schedule.length);
    }

    private int[][][] localSearch(int[][][] schedule) {
        double bestFitness = super.fitness(schedule);
        int[][][] bestSchedule = this.copySchedule(schedule);

        for (int iter = 0; iter < this.localSearchIterations; iter++) {
            int i = random.nextInt(super.numNurses);
            int k = random.nextInt(super.numDays);
            int currentShift = this.getCurrentShift(schedule[i][k]);
            int newShift = this.randomChoiceExcluding(currentShift, super.numShifts);

            int[][][] newSchedule = this.copySchedule(schedule);
            newSchedule[i][k][currentShift] = 0;
            newSchedule[i][k][newShift] = 1;

            if (super.isFeasible(newSchedule)) {
                double newFitness = super.fitness(newSchedule);
                if (newFitness < bestFitness) {
                    bestSchedule = this.copySchedule(newSchedule);
                    bestFitness = newFitness;
                }
            }
        }
        return bestSchedule;
    }

    private int[][][] copySchedule(int[][][] schedule) {
        int[][][] copy = new int[schedule.length][][];
        for (int i = 0; i < schedule.length; i++) {
            copy[i] = new int[schedule[i].length][];
            for (int j = 0; j < schedule[i].length; j++) {
                copy[i][j] = Arrays.copyOf(schedule[i][j], schedule[i][j].length);
            }
        }
        return copy;
    }

    private int getCurrentShift(int[] shifts) {
        for (int s = 0; s < shifts.length; s++) {
            if (shifts[s] == 1) {
                return s;
            }
        }
        return -1;
    }

    private int randomChoiceExcluding(int exclude, int range) {
        int choice;
        do {
            choice = random.nextInt(range);
        } while (choice == exclude);
        return choice;
    }
    
    
    @Override
    public void run(){
        for(int generation = 0; generation < super.generations; generation++){
            Individual[] newPopulation = new Individual[super.populationSize];
            int count = 0;
            Individual[] parents = super.selectParents();
            while(count < super.populationSize){
                int[][][] child = super.crossover(parents);
                
                // dopo la mutazione c'è la local search
                this.mutate(child);
                
                if(super.isFeasible(child)){
                    newPopulation[count] = new Individual(child, super.fitness(child));
                    count++;
                }
            }
            Arrays.sort(newPopulation);
            super.population = Arrays.copyOfRange(newPopulation, 0, super.populationSize);

            super.printGenerationInfo(generation, super.population);
        }
        
        Individual bestIndividual = this.population[0];
        super.bestSchedule = bestIndividual.getSchedule();
        super.bestFitness = bestIndividual.getFitness();
        printBestSolution(this.bestSchedule, this.bestFitness);
    }
    
    
    public static void main(String[] args) throws IOException{
        long startTime = System.currentTimeMillis();
        String filename = "1.nsp";
        NSPGeneticAlgorithmLocalSearch gals = new NSPGeneticAlgorithmLocalSearch(filename, 15, 200, 0.3, 100, 1);
        gals.run();
        long endTime = System.currentTimeMillis();
        double duration = (endTime - startTime) / 1000.0;
        System.out.println("Elapsed time: " + duration + " seconds");
    }
}
