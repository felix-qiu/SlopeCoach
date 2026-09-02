import Foundation
import Observation

@MainActor
@Observable
final class RiderSelectionViewModel {
    let videoSource: RiderVideoSource
    let sourcePixelSize = CGSize(width: 1400, height: 933)
    let duration: TimeInterval

    private(set) var currentTime: TimeInterval
    private(set) var isPlaying = false
    private(set) var isScrubbing = false
    private(set) var selectionState: RiderSelectionState = .idle
    private(set) var selectedRider: RiderSelectionResult?
    private(set) var currentMask: RiderMask?
    private(set) var tapPoint: NormalizedVideoPoint?
    private(set) var pulseID = UUID()
    private(set) var errorMessage: String?

    private let segmenter: any RiderSegmenter
    private var selectionTask: Task<Void, Never>?
    private var playbackTask: Task<Void, Never>?
    private var maskTask: Task<Void, Never>?

    init(
        videoSource: RiderVideoSource = .asset(name: "skiHero"),
        duration: TimeInterval = 58,
        initialTime: TimeInterval = 12,
        segmenter: any RiderSegmenter = MockRiderSegmenter()
    ) {
        self.videoSource = videoSource
        self.duration = duration
        self.currentTime = min(max(initialTime, 0), duration)
        self.segmenter = segmenter
    }

    var canContinue: Bool {
        selectionState == .selected && selectedRider != nil
    }

    var formattedCurrentTime: String {
        RiderSelectionTimeFormatter.string(from: currentTime)
    }

    var formattedDuration: String {
        RiderSelectionTimeFormatter.string(from: duration)
    }

    var formattedSelectionTime: String? {
        selectedRider.map { RiderSelectionTimeFormatter.string(from: $0.timestamp) }
    }

    @discardableResult
    func selectRider(at point: NormalizedVideoPoint) -> Task<Void, Never>? {
        guard selectionState == .idle || selectionState == .failed else { return nil }

        stopPlayback()
        selectionTask?.cancel()
        maskTask?.cancel()
        tapPoint = point
        pulseID = UUID()
        errorMessage = nil
        currentMask = nil
        selectionState = .selecting

        let input = RiderSelectionInput(
            videoSource: videoSource,
            timestamp: currentTime,
            frameIndex: Int((currentTime * 30).rounded()),
            point: point
        )

        let task = Task { [weak self, segmenter] in
            do {
                let result = try await segmenter.selectRider(input: input)
                guard !Task.isCancelled, let self else { return }
                self.selectedRider = result
                self.currentMask = result.initialMask
                self.selectionState = .selected
            } catch is CancellationError {
                return
            } catch {
                guard !Task.isCancelled, let self else { return }
                self.selectedRider = nil
                self.currentMask = nil
                self.errorMessage = error.localizedDescription
                self.selectionState = .failed
            }
        }
        selectionTask = task
        return task
    }

    func changeSelection() {
        clearSelection(keepingTime: true)
    }

    func tryAgain() {
        clearSelection(keepingTime: true)
    }

    func reselectAfterTrackingLoss() {
        clearSelection(keepingTime: true)
    }

    func markTrackingLost() {
        guard selectionState == .selected else { return }
        currentMask = nil
        selectionState = .lost
    }

    func beginScrubbing() {
        stopPlayback()
        isScrubbing = true
        currentMask = nil
        maskTask?.cancel()
    }

    func scrub(to time: TimeInterval) {
        currentTime = clampedTime(time)
        if !isScrubbing {
            refreshMaskForCurrentTime()
        }
    }

    func endScrubbing() {
        isScrubbing = false
        refreshMaskForCurrentTime()
    }

    func togglePlayback() {
        isPlaying ? stopPlayback() : startPlayback()
    }

    func stopPlayback() {
        isPlaying = false
        playbackTask?.cancel()
        playbackTask = nil
    }

    private func startPlayback() {
        guard currentTime < duration else {
            currentTime = 0
            return startPlayback()
        }

        isPlaying = true
        playbackTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(50))
                guard !Task.isCancelled, let self else { return }

                self.currentTime = self.clampedTime(self.currentTime + 0.05)
                self.refreshMaskForCurrentTime()
                if self.currentTime >= self.duration {
                    self.stopPlayback()
                    return
                }
            }
        }
    }

    private func clearSelection(keepingTime: Bool) {
        selectionTask?.cancel()
        maskTask?.cancel()
        selectedRider = nil
        currentMask = nil
        tapPoint = nil
        errorMessage = nil
        selectionState = .idle
        if !keepingTime {
            currentTime = 0
        }
    }

    private func refreshMaskForCurrentTime() {
        guard !isScrubbing,
              selectionState == .selected,
              let session = selectedRider?.session else {
            currentMask = nil
            return
        }

        let requestedTime = currentTime
        maskTask?.cancel()
        maskTask = Task { [weak self] in
            let mask = await session.mask(at: requestedTime)
            guard !Task.isCancelled, let self, abs(self.currentTime - requestedTime) < 0.05 else {
                return
            }
            self.currentMask = mask
        }
    }

    private func clampedTime(_ time: TimeInterval) -> TimeInterval {
        min(max(time, 0), duration)
    }
}

