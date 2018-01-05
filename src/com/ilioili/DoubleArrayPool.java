package com.ilioili;

import java.util.Arrays;
import java.util.LinkedList;

/**
 * Created by ilioili on 2018/1/3.
 */
public class DoubleArrayPool {
    private static LinkedList<double[]> cacheList = new LinkedList<>();

    public static double[] get(int lenght) {
        double[] result = null;
        for (double[] doubleArray : cacheList) {
            if (doubleArray.length == lenght) {
                result = doubleArray;
                break;
            }
        }
        if (null == result) {
            return new double[lenght];
        } else {
            cacheList.remove(result);
        }
        return result;
    }

    public static void recycle(double[] array){
        Arrays.fill(array, 0);
        cacheList.add(array);
    }
}
