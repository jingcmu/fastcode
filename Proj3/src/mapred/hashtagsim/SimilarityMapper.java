package mapred.hashtagsim;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.VIntWritable;
import org.apache.hadoop.mapreduce.Mapper;

public class SimilarityMapper extends Mapper<LongWritable, Text, VIntWritable, Text> {

	// Map<String, Integer> jobFeatures = null;
	String tag = null;
	Map<String, Integer> tagFeatures = null;

	/**
	 * We compute the inner product of feature vector of every hashtag with that
	 * of the specific tag
	 */
	@Override
	protected void map(LongWritable key, Text value, Context context)
			throws IOException, InterruptedException {
		Configuration configuration = context.getConfiguration();
		FileSystem fs = FileSystem.get(configuration);
		FSDataInputStream in = null;
		BufferedReader br = null;
		try {

			in = fs.open(new Path(configuration.get("input")));
			br = new BufferedReader(new InputStreamReader(in));

			String tmpStr = "";

			// each hash tag feature vector from value
			String[] hashtag_featureVector1 = value.toString().split("\\s+", 2);
			String hashtag1 = hashtag_featureVector1[0];
			Map<String, Integer> features1 = parseFeatureVector(hashtag_featureVector1[1]);

			while ((tmpStr = br.readLine()) != null) {

				// each hash tag feature vector from the file
				String[] hashtag_featureVector2 = null;
				String hashtag2 = null;
				hashtag_featureVector2 = tmpStr.split("\\s+", 2);
				hashtag2 = hashtag_featureVector2[0];
				Map<String, Integer> features2 = parseFeatureVector(hashtag_featureVector2[1]);

				Integer similarity = computeInnerProduct(features1, features2);

				// ignore 0 similarity
				if (similarity == 0) {
					continue;
				}

				context.write(new VIntWritable(similarity), new Text(hashtag1
						+ "\t" + hashtag2));				
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * This function is ran before the mapper actually starts processing the
	 * records, so we can use it to setup the tag feature vector.
	 * 
	 * Loads the feature vector for hashtag this tag into mapper's memory
	 */
	@Override
	protected void setup(Context context) {
	}

	/**
	 * De-serialize the feature vector into a map
	 * 
	 * @param featureVector
	 *            The format is "word1:count1;word2:count2;...;wordN:countN;"
	 * @return A HashMap, with key being each word and value being the count.
	 */
	private Map<String, Integer> parseFeatureVector(String featureVector) {
		Map<String, Integer> featureMap = new HashMap<String, Integer>();
		String[] features = featureVector.split(";");
		for (String feature : features) {
			String[] word_count = feature.split(":");
			featureMap.put(word_count[0], Integer.parseInt(word_count[1]));
		}
		return featureMap;
	}

	/**
	 * Computes the dot product of two feature vectors
	 * 
	 * @param featureVector1
	 * @param featureVector2
	 * @return
	 */
	private Integer computeInnerProduct(Map<String, Integer> featureVector1,
			Map<String, Integer> featureVector2) {
		Integer sum = 0;
		for (String word : featureVector1.keySet())
			if (featureVector2.containsKey(word))
				sum += featureVector1.get(word) * featureVector2.get(word);

		return sum;
	}
}
