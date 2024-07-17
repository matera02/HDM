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
public class NSPGeneticAlgorithm extends NSP {
    
    protected int populationSize;
    protected int generations;
    protected double mutationRate;
    protected Individual[] population;
    protected int crossoverType;
    protected int[][][] bestSchedule;
    protected double bestFitness;
    
    public NSPGeneticAlgorithm(String filename, int populationSize, int generations, 
            double mutationRate, int crossoverType) throws IOException {
        super(filename);
        this.populationSize = populationSize;
        this.generations = generations;
        this.mutationRate = mutationRate;
        this.population = new Individual[populationSize];
        this.crossoverType = crossoverType;
        this.initializePopulation();
    }
    
    private void initializePopulation(){
        int count = 0;
        while(count < this.populationSize){
            int[][][] schedule = randomSchedule();
            if(isFeasible(schedule)){
                this.population[count] = new Individual(schedule, fitness(schedule));
                count++;
            }
        }
    }
    
    protected Individual[] selectParents(){
        Individual[] selected = new Individual[4];
        int count = 0;
        while(count < 4){
            Individual[] participants = new Individual[2];
            participants[0] = this.population[random.nextInt(this.populationSize)];
            participants[1] = this.population[random.nextInt(this.populationSize)];
            selected[count] = participants[0].fitness < participants[1].fitness ? participants[0] : participants[1];
            count++;
        }
        return selected;
    }
    
    protected int[][][] crossover(Individual[] parents){
        int[][][] child = new int[super.getNumNurses()][super.getNumDays()][super.getNumShifts()];
        Individual parent1 = parents[random.nextInt(parents.length)];
        Individual parent2 = parents[random.nextInt(parents.length)];
        int crossoverPoint = 0;
        switch(this.crossoverType){
            // Metà colonne di ciascun genitore
            case 1:
                for(int k = 0; k < super.getNumDays(); k++){
                    if(k < super.getNumDays() / 2){
                        for(int i = 0; i < super.getNumNurses(); i++){
                            child[i][k] = parent1.schedule[i][k].clone();
                        }
                    } else {
                        for(int i = 0; i < super.getNumNurses(); i++){
                            child[i][k] = parent2.schedule[i][k].clone();
                        }
                    }
                }
                break;
            // Random crossover point sul giorno
            case 2:
                crossoverPoint = random.nextInt(getNumDays() - 1) + 1;
                for(int i = 0; i < super.getNumNurses(); i++){
                    
                    for(int k = 0; k < crossoverPoint; k++){
                        child[i][k] = parent1.schedule[i][k].clone();
                    }
                    for(int k = crossoverPoint; k < getNumDays(); k++){
                        child[i][k] = parent2.schedule[i][k].clone();
                    }
                }
                break;
            case 3:
                // Random crossover point sul numero di infermieri
                crossoverPoint = random.nextInt(getNumNurses() - 1) + 1;
                for(int i = 0; i < crossoverPoint; i++){
                    for(int k = 0; k < super.getNumDays(); k++){
                        child[i][k] = parent1.schedule[i][k].clone();
                    }
                }
                for(int i = crossoverPoint; i < super.getNumNurses(); i++){
                    for(int k = 0; k < super.getNumDays(); k++){
                        child[i][k] = parent2.schedule[i][k].clone();
                    }
                }
                break;
            default:
                break;
        }
        return child;
    }
    
    /*Questo metodo verrà sovrascritto dalla sottoclasse in cui
      aggiungo l'algoritmo di ricerca locale nell'implementazione
    */
    protected void mutate(int[][][] schedule){
        if(random.nextDouble() < this.mutationRate){
            int i = random.nextInt(super.getNumNurses());
            int k = random.nextInt(super.getNumDays());
            int s = random.nextInt(super.getNumShifts());
            Arrays.fill(schedule[i][k], 0);
            schedule[i][k][s] = 1;
        }
    }
    
    
    protected void printGenerationInfo(int generation, Individual[] population) {
        double bestFitness = population[0].fitness;
        double avgFitness = Arrays.stream(population).mapToDouble(ind -> ind.fitness).average().orElse(0);
        System.out.println("Generation " + (generation + 1) + ": Best Fitness = " + bestFitness + ", Average Fitness = " + avgFitness);
    }
    
    
    @Override
    public void run() {
        for(int generation = 0; generation < this.generations; generation++){
            Individual[] newPopulation = new Individual[this.populationSize];
            int count = 0;
            Individual[] parents = this.selectParents();
            while(count < this.populationSize){
                int[][][] child = this.crossover(parents);
                this.mutate(child);
                if(super.isFeasible(child)){
                    newPopulation[count] = new Individual(child, super.fitness(child));
                    count++;
                }
            }
            Arrays.sort(newPopulation);
            this.population = Arrays.copyOfRange(newPopulation, 0, this.populationSize);

            this.printGenerationInfo(generation, this.population);
        }
        
        Individual bestIndividual = this.population[0];
        this.bestSchedule = bestIndividual.getSchedule();
        this.bestFitness = bestIndividual.getFitness();
        printBestSolution(this.bestSchedule, this.bestFitness);
        
    }
    
    
    public int[][][] getBestSchedule(){
        return bestSchedule;
    }
    
    
    public double getBestFitness(){
        return bestFitness;
    }
    
    
    protected class Individual implements Comparable<Individual> {
        private int[][][] schedule;
        private double fitness;

        Individual(int[][][] schedule, double fitness) {
            this.schedule = schedule;
            this.fitness = fitness;
        }

        @Override
        public int compareTo(Individual other) {
            return Double.compare(this.fitness, other.fitness);
        }
        
        
        public int[][][] getSchedule(){
            return this.schedule;
        }
        
        public double getFitness(){
            return this.fitness;
        }
    }
    
    
    public static void main(String[] args) throws IOException{
        long startTime = System.currentTimeMillis();
        String filename = "1.nsp";
        NSPGeneticAlgorithm ga = new NSPGeneticAlgorithm(filename, 15, 200, 0.3, 1);
        ga.run();
        long endTime = System.currentTimeMillis();
        double duration = (endTime - startTime) / 1000.0;
        System.out.println("Elapsed time: " + duration + " seconds");
    }
    
}
