import Foundation

struct MockRiderSegmenter: RiderSegmenter {
    struct Configuration: Sendable {
        let riderCenter: NormalizedVideoPoint
        let horizontalRadius: CGFloat
        let verticalRadius: CGFloat
        let processingDelay: Duration
        let propagatedFrameWindow: TimeInterval

        static let demo = Configuration(
            riderCenter: NormalizedVideoPoint(x: 0.75, y: 0.59)!,
            horizontalRadius: 0.18,
            verticalRadius: 0.29,
            processingDelay: .milliseconds(450),
            propagatedFrameWindow: 0.75
        )
    }

    let configuration: Configuration

    init(configuration: Configuration = .demo) {
        self.configuration = configuration
    }

    func selectRider(input: RiderSelectionInput) async throws -> RiderSelectionResult {
        try await Task.sleep(for: configuration.processingDelay)

        let dx = (input.point.x - configuration.riderCenter.x) / configuration.horizontalRadius
        let dy = (input.point.y - configuration.riderCenter.y) / configuration.verticalRadius
        guard dx * dx + dy * dy <= 1 else {
            throw RiderSegmentationError.riderNotFound
        }

        let selectionID = UUID().uuidString
        let mask = Self.demoRiderMask
        let session = MockRiderTrackingSession(
            selectionID: selectionID,
            anchorTimestamp: input.timestamp,
            anchorMask: mask,
            availableFrameWindow: configuration.propagatedFrameWindow
        )

        return RiderSelectionResult(
            selectionID: selectionID,
            timestamp: input.timestamp,
            initialMask: mask,
            confidence: 0.91,
            session: session
        )
    }

    private static let demoRiderMask: RiderMask = .normalizedPolygon([
        NormalizedVideoPoint(x: 0.76, y: 0.39)!,
        NormalizedVideoPoint(x: 0.80, y: 0.43)!,
        NormalizedVideoPoint(x: 0.81, y: 0.50)!,
        NormalizedVideoPoint(x: 0.86, y: 0.55)!,
        NormalizedVideoPoint(x: 0.84, y: 0.60)!,
        NormalizedVideoPoint(x: 0.79, y: 0.58)!,
        NormalizedVideoPoint(x: 0.78, y: 0.68)!,
        NormalizedVideoPoint(x: 0.84, y: 0.75)!,
        NormalizedVideoPoint(x: 0.81, y: 0.80)!,
        NormalizedVideoPoint(x: 0.73, y: 0.72)!,
        NormalizedVideoPoint(x: 0.68, y: 0.73)!,
        NormalizedVideoPoint(x: 0.65, y: 0.68)!,
        NormalizedVideoPoint(x: 0.71, y: 0.63)!,
        NormalizedVideoPoint(x: 0.72, y: 0.53)!,
        NormalizedVideoPoint(x: 0.70, y: 0.48)!
    ])
}

struct MockRiderTrackingSession: RiderTrackingSession {
    let selectionID: String
    let anchorTimestamp: TimeInterval
    let anchorMask: RiderMask
    let availableFrameWindow: TimeInterval

    func mask(at timestamp: TimeInterval) async -> RiderMask? {
        guard abs(timestamp - anchorTimestamp) <= availableFrameWindow else {
            return nil
        }
        return anchorMask
    }
}

