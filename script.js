function normalize(tensor, prevMin, prevMin) {
	const min = prevMin || tensor.min(),
		max = prevMin || tensor.max(),
		normalisedTensor = tensor.sub(min).div(max.sub(min));
	return normalisedTensor;
}

function denormalize(tensor, prevMin, prevMin) {
	const min = prevMin || tensor.min(),
		max = prevMin || tensor.max(),
		denormalisedTensor = tensor.mul(max.sub(min)).add(min);
	return denormalisedTensor;
}

function plot(points, predictedPoints) {
	const data = {
		values: [points, ...(predictedPoints ? [predictedPoints] : [])],
		series: ["original", ...(predictedPoints ? ["prediction"] : [])],
	};

	const surface = { name: "ICICI Bank stock price prediction" };
	tfvis.render.scatterplot(surface, data, { xLabel: "Open", yLabel: "Close" });
}

async function loadDataset() {
	const data = tf.data.csv("http://127.0.0.1:3001/HDFCBANK.csv", {
		hasHeader: true,
		columnConfigs: {
			Close: {
				isLabel: true,
			},
			Open: {
				isLabel: false,
			},
		},

		configuredColumnsOnly: true,
		// delimWhitespace: false,
	});
	let dataArr = await data.toArray();
	const flattenedData = dataArr.map((item) => ({
		x: item.xs.Open,
		y: item.ys.Close,
	}));

	console.log(flattenedData);

	tf.util.shuffle(flattenedData);

	// plot(flattenedData);

	let features = flattenedData.map((p) => p.x);
	let labels = flattenedData.map((p) => p.y);
	console.log(features);
	let featureTensor = tf.tensor2d(features, [features.length, 1]);
	let labelTensor = tf.tensor2d(labels, [labels.length, 1]);
	let normalisedFeatureTensor = normalize(featureTensor);
	let normalisedLabelTensor = normalize(labelTensor);
	const [trainFeatureTensor, testFeatureTensor] = tf.split(
		normalisedFeatureTensor,
		2
	);
	const [trainLabelTensor, testLabelTensor] = tf.split(
		normalisedLabelTensor,
		2
	);

	const model = tf.sequential();
	model.add(
		tf.layers.dense({
			inputShape: [1],
			units: 1,
			useBias: true,
			activation: "linear",
		})
	);
	model.summary();
	let optimizer = tf.train.sgd(0.1);
	model.compile({ loss: "meanSquaredError", optimizer: optimizer });
	await model.fit(trainFeatureTensor, trainLabelTensor, {
		epochs: 10,
		batchSize: 32,
		validationData: [trainFeatureTensor, trainLabelTensor],
	});
	const testing = await model.evaluate(testFeatureTensor, testLabelTensor);
	console.log(await testing.dataSync());

	let normalisedXs = [];
	while (normalisedXs.length < 1000) {
		var r = Math.random();
		normalisedXs.push(r);
	}
	normalisedXs = tf.tensor2d(normalisedXs, [1000, 1]);
	const normalisedYs = model.predict(normalisedXs);
	featureTensor.min().print();
	featureTensor.max().print();

	const xs = denormalize(
		normalisedXs,
		featureTensor.min(),
		featureTensor.max()
	).dataSync();
	const ys = denormalize(
		normalisedYs,
		labelTensor.min(),
		labelTensor.max()
	).dataSync();

	const predictedPoints = Array.from(xs).map((val, ind) => ({
		x: val,
		y: ys[ind],
	}));

	plot(flattenedData, predictedPoints);
}

loadDataset();
