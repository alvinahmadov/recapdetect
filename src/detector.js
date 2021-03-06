import {cv}      from "opencv-wasm";
import {Darknet} from "darknet";
import {
	cvRead,
	cvWrite,
	HSVtoRGB,
	readClassNames,
	cvTranslateError
}                from "./utils.js";

export class Detector {
	static FONT_SCALE = 0.5;
	static TEXT_COLOR = [0, 0, 0, 255];
	
	constructor(weightsPath, configPath, classNamesPath,
	            debug = false) {
		this.classNames = readClassNames(classNamesPath);
		this.darknet = new Darknet({weights: weightsPath, config: configPath, namefile: classNamesPath});
		this.debug = debug;
	}
	
	/**
	 * Draw predicted boxes on the image with labeling
	 * with name and probability
	 *
	 * @param {string} imagePath: Path to image to be predicted
	 * @param {Array} predictions
	 * */
	async drawBox(imagePath, predictions) {
		try {
			const alpha = 255;
			const image = await cvRead(imagePath);
			const [width, height] = [image.cols, image.rows];
			const numClasses = this.classNames.size;
			let clsIdx;
			
			const colors = Array(numClasses)
				.fill({h: 1.0, s: 1.0, v: 1.0})
				.map((value, index) => {
					return value = {h: index / numClasses, s: 1.0, v: 1.0};
				})
				.map((value => HSVtoRGB(value)))
				.shuffle();
			
			predictions.forEach(pred => {
				this.classNames.forEach((v, k) => {if (v.trim() === pred.name.trim()) clsIdx = k;});
				const [x, y, w, h] = [pred.box.x, pred.box.y, pred.box.w, pred.box.h];
				const colorObj = colors[clsIdx];
				const color = new cv.Scalar(colorObj.r, colorObj.g, colorObj.b, alpha);
				const boxText = `${pred.name}: ${pred.prob.toFixed(2)}`;
				const textColor = new cv.Scalar(...Detector.TEXT_COLOR);
				let coord = [
					[x - Math.floor(w / 2), y - Math.floor(h / 2)],
					[x + Math.floor(w / 2), y + Math.floor(h / 2)]
				];
				const [p1, p2] = [new cv.Point(...coord[0]), new cv.Point(...coord[1])];
				const textCoord = new cv.Point(coord[0][0] + 110, coord[0][1] - 15);
				const thickness = 0.6 * (height + width) / 600;
				
				cv.rectangle(image, p1, p2, color, thickness);
				
				cv.rectangle(image, p1, textCoord, color, cv.FILLED);
				cv.putText(
					image,
					boxText,
					new cv.Point(coord[0][0], coord[0][1] - 2),
					cv.FONT_HERSHEY_SIMPLEX,
					Detector.FONT_SCALE,
					textColor,
					1.5,
					cv.FILLED,
					false
				);
			})
			return image;
		} catch (e) {
			console.error(cvTranslateError(e));
		}
	}
	
	/**
	 * @param {string} imagePath
	 * @param {string|null} savePath
	 * */
	async detect(imagePath, savePath = null) {
		try {
			const predictions = this.darknet.detect(imagePath);
			const image = await this.drawBox(imagePath, predictions);
			
			if (this.debug) {
				console.log('Predictions:');
				console.log(predictions);
			}
			
			try {
				if (savePath !== null && savePath !== '') {
					if (!savePath.endsWith('.png'))
						savePath += '.png';
					cvWrite(image, savePath).then(() => console.log(`Image saved at ${savePath}`));
				}
			} catch (e) {
				console.error(e);
			}
			
			image.delete();
		} catch (e) {
			console.error(e);
		}
	}
}
