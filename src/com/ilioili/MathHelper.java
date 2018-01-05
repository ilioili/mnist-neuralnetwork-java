package com.ilioili;

import java.util.List;
import java.util.Random;

/**
 * Created by ilioili on 2018/1/3.
 */
public class MathHelper {
    public static void mutiply(double[] array1, double[] array2, double[] resultArray) {
        assert array1.length == array2.length;
        int length = array1.length;
        for (int i = 0; i < length; i++) {
            resultArray[i] = array1[i] * array2[i];
        }
    }

    public static void plus(double[] array1, double[] array2, double[] resultArray) {
        assert array1.length == array2.length;
        int length = array1.length;
        for (int i = 0; i < length; i++) {
            resultArray[i] = array1[i] + array2[i];
        }
    }

    public static double sigmod(double x) {
        return 1.0d / (1 + Math.exp(-x));
    }

    public static double sigmodPrime(double x) {
        return sigmod(x) * (1 - sigmod(x));
    }

    private static Random random = new Random(System.currentTimeMillis());

    public static void random(double[] array) {

        for (int i = 0; i < array.length; i++) {
            array[i] = random.nextDouble();
        }
    }

    public static void set(double[] array, double value) {
        for (int i = 0; i < array.length; i++)
            array[i] = value;
    }

}
