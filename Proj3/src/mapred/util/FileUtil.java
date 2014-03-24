package mapred.util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import com.sun.corba.se.spi.orbutil.fsm.Input;

import mapred.filesystem.CommonFileOperations;

public class FileUtil {
	static Configuration conf;
	static FileSystem fs;

	public static void setConfiguration(Configuration c) {
		conf = c;
		try {
			fs = FileSystem.get(conf);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	static {
		setConfiguration(new Configuration());
	}

	public static void save(String str, String filename) throws IOException {
		OutputStream out = write(filename);
		BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(out));
		writer.write(str);
		writer.close();
	}

	// Only loads the first line
	public static String load(String filename) throws IOException {
		InputStream is = read(filename);
		BufferedReader reader = new BufferedReader(new InputStreamReader(is));
		String str = reader.readLine();
		reader.close();
		return str;
	}

	// Saves to temp dir
	public static void saveTmp(String str, String filename) throws IOException {
		save(str, getTmpDir() + "/" + filename);
	}

	// Saves to temp dir
	public static String loadTmp(String filename) throws IOException {
		return load(getTmpDir() + "/" + filename);
	}

	// Saves to temp dir
	public static void saveInt(int i, String name) {
		try {
			saveTmp(Integer.toString(i), name);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	// Loads from temp dir
	public static Integer loadInt(String name) {
		try {
			return Integer.parseInt(loadTmp(name));
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}
	}

	// Saves to temp dir
	public static void saveLong(long i, String name) {
		try {
			saveTmp(Long.toString(i), name);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	// Loads from temp dir
	public static Long loadLong(String name) {
		try {
			return Long.parseLong(loadTmp(name));
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}
	}

	// Saves to temp dir
	public static void saveDouble(Double i, String name) {
		try {
			saveTmp(Double.toString(i), name);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	// Loads from temp dir
	public static Double loadDouble(String name) {
		try {
			return Double.parseDouble(loadTmp(name));
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}
	}

	// Loads all lines
	public static InputLines loadLines(String filename) throws IOException {
		InputStream is = read(filename);
		return new InputLines(is);
	}

	public static OutputStream write(String filename) throws IOException {
		CommonFileOperations.deleteIfExists(filename);
		return fs.create(new Path(filename));
	}

	public static InputStream read(String filename) throws IOException {
		return fs.open(new Path(filename));
	}

	public static String getTmpDir() {
		return conf.get("hadoop.tmp.dir");
	}

	public static FileSystem getFS() {
		return fs;
	}

	public static void testFileOperation() throws IOException {
		FileUtil.save("abc\n", "data/abc");
		String str = FileUtil.load("data/abc");
		System.out.println(str);
		for (String s : FileUtil.loadLines("data/abstract.tiny.test"))
			System.out.println(s);
	}

	public static void copyFile(String inputFilename, String outputFilename) {
		FSDataInputStream in = null;
		FSDataOutputStream out = null;
		byte buffer[] = new byte[256];
		try {
			in = fs.open(new Path(inputFilename));
			out = fs.create(new Path(outputFilename));
			int bytesRead = 0;
			while ((bytesRead = in.read(buffer)) > 0) {
				out.write(buffer, 0, bytesRead);
			}
		} catch (IOException e) {
			e.printStackTrace();
			System.out.println("Error while copying file");
		} finally {
			try {
				if (in != null)
					in.close();

				if (out != null)
					out.close();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}

	public static void joinFiles(String inputFilename1, String inputFilename2,
			String outputFilename) {
		FSDataInputStream in1 = null;
		FSDataInputStream in2 = null;
		BufferedReader br1 = null;
		BufferedReader br2 = null;
		FSDataOutputStream out = null;
		BufferedWriter bw = null;
		try {

			in1 = fs.open(new Path(inputFilename1));
			in2 = fs.open(new Path(inputFilename2));
			br1 = new BufferedReader(new InputStreamReader(in1));
			br2 = new BufferedReader(new InputStreamReader(in2));

			out = fs.create(new Path(outputFilename));
			bw = new BufferedWriter(new OutputStreamWriter(out));

			String tmpStr = "";

			ArrayList<String> strList = new ArrayList<String>();
			while ((tmpStr = br1.readLine()) != null) {
				strList.add(tmpStr);
			}

			for (String str1 : strList) {
				bw.write(str1+"\t");
				for (String str2 : strList) {
					bw.write(str2+"\t");
				}
				bw.newLine();
			}

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		} finally {
			try {
				if (br1 != null) {
					br1.close();
				}
				if (br2 != null) {
					br2.close();
				}
				if (bw != null) {
					bw.close();
				}
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
}
