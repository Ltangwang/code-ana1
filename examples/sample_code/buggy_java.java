// Sample Java code with intentional bugs for testing

public class BuggyJava {
    
    // Null pointer exception possible - BUG!
    public static String getFirstChar(String str) {
        return str.substring(0, 1);
    }
    
    // Array index out of bounds - BUG!
    public static int getElement(int[] arr, int index) {
        return arr[index];
    }
    
    // Resource leak - BUG!
    public static String readFile(String filename) throws Exception {
        java.io.FileReader fr = new java.io.FileReader(filename);
        java.io.BufferedReader br = new java.io.BufferedReader(fr);
        return br.readLine();
    }
    
    // Integer overflow - BUG!
    public static int multiply(int a, int b) {
        return a * b;
    }
    
    // Potential infinite loop - BUG!
    public static void processUntilZero(int[] values) {
        int i = 0;
        while (values[i] != 0) {
            System.out.println(values[i]);
            // Forgot to increment i
        }
    }
}

