package com.ilioili;

import java.util.Random;

public class Main {


    public static void main(String[] args) {
        // write your code here
        MnistDataLoader.loadData();

//        test();

        testImage();

//        debugNetwork();

    }

    private static void test() {
        //y = 3X+4Y+6Z+7W;
        int i = 0;
        NeuralNetwork neuralNetwork = new NeuralNetwork(10, 4, 8, 4);
        double cost = neuralNetwork.getLoss();
        Random random = new Random(0);
        double[] trainData = new double[]{random.nextDouble(), random.nextDouble(), random.nextDouble(), random.nextDouble()};
        double label = trainData[0] * 3 + trainData[1] * 4 + trainData[2] * 6 + trainData[3] * 7;
        double[] labelData = new double[]{trainData[0] * 3, trainData[1] * 4, trainData[2] * 6, trainData[3] * 7};
        while (true) {
            trainData = new double[]{random.nextDouble(), random.nextDouble(), random.nextDouble(), random.nextDouble()};
            labelData = new double[]{trainData[0] * 3, trainData[1] * 4, trainData[2] * 6, trainData[3] * 7};
            for (int index = 0; index < labelData.length; index++) {
                labelData[index] = MathHelper.sigmod(labelData[index]);
            }
            neuralNetwork.feedForward(trainData);
            neuralNetwork.backPropergation(labelData);
            if ((i++) % 100000 == 0)
                System.out.println("i=" + i + " cost is down:" + (cost > neuralNetwork.getLoss()) + " cost:" + neuralNetwork.getLoss());
            cost = neuralNetwork.getLoss();
            if (cost < 1e-7) {
                System.out.println(neuralNetwork.getStructureStr());
                break;
            }
        }
        i = 0;
        System.out.println(">>>>>>>>>>>>>>>>>");
        while (true) {
            trainData = new double[]{random.nextDouble(), random.nextDouble(), random.nextDouble(), random.nextDouble()};
            labelData = new double[]{trainData[0] * 3, trainData[1] * 4, trainData[2] * 6, trainData[3] * 7};
            for (int index = 0; index < labelData.length; index++) {
                labelData[index] = MathHelper.sigmod(labelData[index]);
            }
            neuralNetwork.feedForward(trainData);
            neuralNetwork.computeCost(labelData);
            System.out.println("i=" + i + " cost is down:" + (cost > neuralNetwork.getLoss()) + " cost:" + neuralNetwork.getLoss());
            cost = neuralNetwork.getLoss();
            if (i++ > 100) {
                break;
            }
        }
    }

    private static void debugNetwork() {
        double[] outputLabels = {1, 1, 1};
        double[] inputData = {1};
        NeuralNetwork neuralNetwork = new NeuralNetwork(1, inputData.length, 1, outputLabels.length);
        int i = 0;
        double cost = neuralNetwork.getLoss();

        while (true) {
            neuralNetwork.feedForward(inputData);
            neuralNetwork.backPropergation(outputLabels);
            if ((i++) % 10000 == 0)
                System.out.println("i=" + i + " cost is down:" + (cost > neuralNetwork.getLoss()) + " cost:" + neuralNetwork.getLoss());
            if (neuralNetwork.getLoss() < 0.00001d) {
                System.out.println("\ni=" + i + "  " + neuralNetwork.getStructureStr());
                break;
            }
            cost = neuralNetwork.getLoss();
        }
    }

    private static void testImage() {

        NeuralNetwork neuralNetwork = new NeuralNetwork(0.5, 784, 30, 10);
        int i = 0;
        int batchSize = 1000;
        while (true) {
            double[] trainData = MnistDataLoader.getImageData(i % batchSize);
            double[] labelData = MnistDataLoader.getImageLabel(i % batchSize);
            neuralNetwork.feedForward(trainData);
            neuralNetwork.backPropergation(labelData);


//            if (i > 60000*100) {
//                MnistDataLoader.printImage(trainData);
//                System.out.println(ArrayUtil.toString(labelData, "0.0"));
//                System.out.println(ArrayUtil.toString(neuralNetwork.getOutputLayer(), "0.0"));
//                System.out.println("\n\n");
//
//            }
            if (i > 300000) {
//                MnistDataLoader.printImage(trainData);
                System.out.println(match(labelData, neuralNetwork.getOutputLayer())?"正确":"错误");
                System.out.print(ArrayUtil.toString(labelData, "0.0"));
                System.out.println(ArrayUtil.toString(neuralNetwork.getOutputLayer(), "0.0"));
//                System.out.println(neuralNetwork.getStructureStr());
                System.out.println("\n");
            }
            i++;
            if (i > 300000 + 10) break;
        }
    }

    private static boolean match(double[] d1, double[] d2) {
        for (int i = 0; i < d1.length; i++) {
            if (Math.abs(d1[i] - d2[i]) > 0.1) {
                return false;
            }
        }
        return true;
    }

    /**

     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//0
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//1
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//2
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//3
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//4
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//5
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//6
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//7
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//8
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//9
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//0
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//1
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//2
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//3
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//4
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//5
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//6
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//7
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//8
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//9
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//0
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//1
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//2
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//3
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//4
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//5
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//6
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//7
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//8
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//9
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//0
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//1
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//2
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//3
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//4
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//5
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//6
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//7
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//8
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//9
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//0
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//1
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//2
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//3
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//4
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//5
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//6
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//7
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//8
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//9
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//0
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//1
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//2
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//3
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//4
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//5
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//6
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//7
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//8
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//9
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//0
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//1
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//2
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//3
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//4
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//5
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//6
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//7
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//8
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//9
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//0
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//1
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//2
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//3
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//4
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//5
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//6
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//7
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//8
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//9
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//0
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//1
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//2
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//3
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//4
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//5
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//6
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//7
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//8
     1, 2, 3, 4, 5, 6, 7, 8, 9, 0,//9
     */

}
