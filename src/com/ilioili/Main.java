package com.ilioili;

import java.text.DecimalFormat;

public class Main {


    public static void main(String[] args) {
        MnistDataLoader.loadData();
        testImage();

    }

    private static void testImage() {

        NeuralNetwork neuralNetwork = new NeuralNetwork(0.05, 784, 30, 10);
        int i = 0;
        int batchSize = 50000;
        int trainTimes = 30000;

        while (true) {
            int index = i % batchSize;
            double[] trainData = MnistDataLoader.getImageData(index);
            double[] labelData = MnistDataLoader.getImageLabel(index);
            neuralNetwork.feedForward(trainData);
            neuralNetwork.backPropergation(labelData);
            if (i == batchSize * trainTimes + batchSize) {
                break;
            }
            if (i % batchSize == 0) {
                test(neuralNetwork);
            }
            i++;
        }

    }

    private static DecimalFormat decimalFormat = new DecimalFormat("0.00");

    private static void test(NeuralNetwork neuralNetwork) {
        int start = 50000;
        int end = start + 10000;
        int accuracy = 0;
        int index = start;
        while (true) {
            double[] trainData = MnistDataLoader.getImageData(index);
            double[] labelData = MnistDataLoader.getImageLabel(index);
            neuralNetwork.feedForward(trainData);
            boolean match = MnistDataLoader.label[index] == MathUtil.max(neuralNetwork.getOutputLayer());
            if (match) accuracy++;
//            System.out.println(match ? "正确" : "错误");
//            System.out.println("Label     :" + ArrayPrintUtil.toString(labelData, "0.0"));
//            System.out.println("Prediction:" + ArrayPrintUtil.toString(neuralNetwork.getOutputLayer(), "0.0"));
//            if (!match) MnistDataLoader.printImage(trainData);
            index++;
            if (index == end) {
                System.out.println("正确率:" + decimalFormat.format(accuracy * 1d / (end - start) * 100) + "%\n");
//                System.out.println(neuralNetwork.getStructureStr());
                break;
            }
        }
//        System.out.println(neuralNetwork.getStructureStr());
    }


}
