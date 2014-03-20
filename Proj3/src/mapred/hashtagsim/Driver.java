package mapred.hashtagsim;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Iterator;
import java.util.LinkedList;

import mapred.job.Optimizedjob;
import mapred.util.FileUtil;
import mapred.util.InputLines;
import mapred.util.SimpleParser;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;

public class Driver {

	public static void main(String args[]) throws Exception {
		SimpleParser parser = new SimpleParser(args);

		String input = parser.get("input");
		String output = parser.get("output");
		String tmpdir = parser.get("tmpdir");

		getJobFeatureVector(input, tmpdir + "/job_feature_vector");
		LinkedList<String> jobFeatureList = loadJobFeatureVector(tmpdir
				+ "/job_feature_vector");


		getHashtagFeatureVector(input, tmpdir + "/feature_vector");

		for (Iterator<String> iterator = jobFeatureList.iterator(); iterator
				.hasNext();) {
			String jobFeatureVector = (String) iterator.next();
			System.out.println("Job feature vector: " + jobFeatureVector);

			getHashtagSimilarities(jobFeatureVector, tmpdir + "/feature_vector",
					output);
		}
	}

	/**
	 * Computes the word cooccurrence counts for hashtag #job
	 * 
	 * @param input
	 *            The directory of input files. It can be local directory, such
	 *            as "data/", "/home/ubuntu/data/", or Amazon S3 directory, such
	 *            as "s3n://myawesomedata/"
	 * @param output
	 *            Same format as input
	 * @throws IOException
	 * @throws ClassNotFoundException
	 * @throws InterruptedException
	 */
//	private static void getJobFeatureVector(String input, String output)
//			throws IOException, ClassNotFoundException, InterruptedException {
//		Optimizedjob job = new Optimizedjob(new Configuration(), input, output,
//				"Get feature vector for hashtag #Job");
//
//		job.setClasses(JobMapper.class, JobReducer.class, null);
//		job.setMapOutputClasses(Text.class, Text.class);
//		job.setReduceJobs(1);
//
//		job.run();
//	}

	// modified version
	private static void getJobFeatureVector(String input, String output)
			throws IOException, ClassNotFoundException, InterruptedException {
		Optimizedjob job = new Optimizedjob(new Configuration(), input, output,
				"Get feature vector for all hashtags");

		job.setClasses(HashtagMapper.class, HashtagReducer.class, null);
		job.setMapOutputClasses(Text.class, Text.class);
		job.setReduceJobs(1);

		job.run();
	}
	/**
	 * Loads the computed word cooccurrence count for hashtag #job from disk.
	 * 
	 * @param dir
	 * @return
	 * @throws IOException
	 */
//	private static String loadJobFeatureVector(String dir) throws IOException {
//		// Since there'll be only 1 reducer that process the key "#job", result
//		// will be saved in the first result file, i.e., part-r-00000
//		String job_featureVector = FileUtil.load(dir + "/part-r-00000");
//
//		// The feature vector looks like "#job word1:count1;word2:count2;..."
//		String featureVector = job_featureVector.split("\\s+", 2)[1];
//		return featureVector;
//	}
	private static LinkedList<String> loadJobFeatureVector(String dir) throws IOException {
		String filename = dir + "/part-r-00000";
		LinkedList<String> tags = new LinkedList<String>();
		
		//String job_featureVector = FileUtil.load(dir + "/part-r-00000");
		Iterator<String> it = FileUtil.loadLines(filename).iterator();
		
		while (it.hasNext()){
			String tag = it.next();
			tags.add(tag.split("\\s+", 2)[1]);
		}
		
		// The feature vector looks like "#job word1:count1;word2:count2;..."
		return tags;
	}
	/**
	 * Same as getJobFeatureVector, but this one actually computes feature
	 * vector for all hashtags.
	 * 
	 * @param input
	 * @param output
	 * @throws Exception
	 */
	private static void getHashtagFeatureVector(String input, String output)
			throws Exception {
		Optimizedjob job = new Optimizedjob(new Configuration(), input, output,
				"Get feature vector for all hashtags");
		job.setClasses(HashtagMapper.class, HashtagReducer.class, null);
		job.setMapOutputClasses(Text.class, Text.class);
		job.run();
	}

	/**
	 * When we have feature vector for both #job and all other hashtags, we can
	 * use them to compute inner products. The problem is how to share the
	 * feature vector for #job with all the mappers. Here we're using the
	 * "Configuration" as the sharing mechanism, since the configuration object
	 * is dispatched to all mappers at the beginning and used to setup the
	 * mappers.
	 * 
	 * @param jobFeatureVector
	 * @param input
	 * @param output
	 * @throws IOException
	 * @throws ClassNotFoundException
	 * @throws InterruptedException
	 */
//	private static void getHashtagSimilarities(String jobFeatureVector,
//			String input, String output) throws IOException,
//			ClassNotFoundException, InterruptedException {
//		// Share the feature vector of #job to all mappers.
//		Configuration conf = new Configuration();
//		conf.set("jobFeatureVector", jobFeatureVector);
//
//		Optimizedjob job = new Optimizedjob(conf, input, output,
//				"Get similarities between #job and all other hashtags");
//		job.setClasses(SimilarityMapper.class, null, null);
//		job.setMapOutputClasses(IntWritable.class, Text.class);
//		job.run();
//	}
	private static void getHashtagSimilarities(String jobFeatureVector,
			String input, String output) throws IOException,
			ClassNotFoundException, InterruptedException {
		// Share the feature vector of #job to all mappers.
		Configuration conf = new Configuration();
		conf.set("jobFeatureVector", jobFeatureVector);

		Optimizedjob job = new Optimizedjob(conf, input, output,
				"Get similarities between any two hashtags");
		job.setClasses(SimilarityMapper.class, null, null);
		job.setMapOutputClasses(IntWritable.class, Text.class);
		job.run();
	}
}
